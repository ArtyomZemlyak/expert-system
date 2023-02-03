#!/usr/bin/env python

import os
import logging

logerror = logging.error
loginfo = logging.info
import ast
import threading

import pyarrow
import pyarrow.flight

import scripts.utils as iu


class ArrowFlightServer(pyarrow.flight.FlightServerBase):
    def __init__(
        self,
        host="0.0.0.0",
        port="8815",  # Check bottom. ARROW_SERVER_PORT bottom.
        location=None,
        cert_chain_file_path=None,
        private_key_file_path=None,
        verify_client=False,  # True - ?
        root_certificates=None,
        auth_handler=None,
        **kwargs,
    ):

        try:
            ARROW_SERVER_PORT = os.environ["ARROW_SERVER_PORT"]
            if ARROW_SERVER_PORT != "":
                port = ARROW_SERVER_PORT
        except Exception as e:
            logerror(" | CANT READ ARROW_SERVER_PORT! USING INSTEAD: {port}")

        tls_certificates = []
        scheme = "grpc"  # "grpc+tcp"
        if cert_chain_file_path != None and private_key_file_path != None:
            scheme = "grpc+tls"
            with open(cert_chain_file_path, "rb") as cert_file:
                tls_cert_chain = cert_file.read()
            with open(private_key_file_path, "rb") as key_file:
                tls_private_key = key_file.read()
            tls_certificates.append((tls_cert_chain, tls_private_key))

        location = f"{scheme}://{host}:{port}"
        loginfo(f"| INIT ARROW FLIGHT SERVER: {location}")

        super(ArrowFlightServer, self).__init__(
            location, auth_handler, tls_certificates, verify_client, root_certificates
        )

        self.flights = (
            {}
        )  # tables, where key/ticket_name/table_name <- descriptor <- file_name/name...

        self.host = host
        self.tls_certificates = tls_certificates

    @classmethod
    def descriptor_to_key(self, descriptor):
        return (
            descriptor.descriptor_type.value,
            descriptor.command,
            tuple(descriptor.path or tuple()),
        )

    def _make_flight_info(self, key, descriptor, table):

        if self.tls_certificates:
            location = pyarrow.flight.Location.for_grpc_tls(self.host, self.port)
        else:
            location = pyarrow.flight.Location.for_grpc_tcp(self.host, self.port)

        endpoints = [
            pyarrow.flight.FlightEndpoint(repr(key), [location]),
        ]

        mock_sink = pyarrow.MockOutputStream()
        stream_writer = pyarrow.RecordBatchStreamWriter(mock_sink, table.schema)
        stream_writer.write_table(table)
        stream_writer.close()
        data_size = mock_sink.size()

        return pyarrow.flight.FlightInfo(
            table.schema, descriptor, endpoints, table.num_rows, data_size
        )

    def list_flights(self, context, criteria):
        for key, table in self.flights.items():
            if key[1] is not None:
                descriptor = pyarrow.flight.FlightDescriptor.for_command(key[1])
            else:
                descriptor = pyarrow.flight.FlightDescriptor.for_path(*key[2])

            yield self._make_flight_info(key, descriptor, table)

    def get_flight_info(self, context, descriptor):
        key = ArrowFlightServer.descriptor_to_key(descriptor)  # key - ticket_name
        if key in self.flights:
            table = self.flights[key]
            return self._make_flight_info(key, descriptor, table)
        raise KeyError("Flight not found.")

    def do_put(self, context, descriptor, reader, writer):
        """Method that is called on the client side to send data to the Arrow Server"""
        key = ArrowFlightServer.descriptor_to_key(descriptor)
        loginfo(f" | KEY: {key} READING DATA...")
        self.flights[key] = reader.read_all()
        loginfo(f" | DONE")

    def do_get(self, context, ticket):
        """Method that is called on the client side to read data from the Arrow Server"""
        key = ast.literal_eval(ticket.ticket.decode())
        if key not in self.flights:
            loginfo(f"| KEY: {key} NOT IN FLIGHTS !")
            return None
        loginfo(f"| KEY: {key} IN FLIGHTS !")
        loginfo(f"| SENDING DATA TO CLIENT...")
        buf = self.flights[key]
        del self.flights[key]
        return pyarrow.flight.RecordBatchStream(buf)

    def list_actions(self, context):
        return [
            ("clear", "Clear the stored flights."),
            ("shutdown", "Shut down this server."),
        ]

    def do_action(self, context, action):
        if action.type == "clear":
            raise NotImplementedError("{} is not implemented.".format(action.type))
        elif action.type == "healthcheck":
            pass
        elif action.type == "shutdown":
            yield pyarrow.flight.Result(pyarrow.py_buffer(b"Shutdown!"))
            # Shut down on background thread to avoid blocking current
            # request
            threading.Thread(target=self._shutdown).start()
        else:
            raise KeyError("Unknown action {!r}".format(action.type))

    def _shutdown(self):
        """Shut down server."""
        self.shutdown()

    def serve_start(self):
        loginfo(f"| START ARROW FLIGHT SERVING ON:{self.port}")
        self.serve()

#!/usr/bin/env python

import os
import logging

logerror = logging.error
loginfo = logging.info

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.flight

# from . import utils as iu


ARROW_SERVER_PORT = os.environ["ARROW_SERVER_PORT"]


class ArrowFlightClient:
    def __init__(
        self,
        ARROW_SERVER_PORT=ARROW_SERVER_PORT,
        cert_chain_file_path=None,
        private_key_file_path=None,
    ):

        self.cert_chain = cert_chain_file_path
        self.private_key = private_key_file_path
        self.data = {}

        """
            tls         - Enable transport-level security
            tls-roots   - Path to trusted TLS certificate(s)
            mtls        - CERTFILE, KEYFILE - Enable transport-level security
            host        - Address or hostname to connect to
        """
        self.host = "arrow"  ########### NEED NAME OF CONTAINER !!!!!!! not localhost or something else -_-
        self.port = int(ARROW_SERVER_PORT)

        self.scheme = "grpc"  # "grpc+tcp"
        self.connection_args = {}

        if self.cert_chain != None and self.private_key != None:
            self.scheme = "grpc+tls"
            with open(self.cert_chain, "rb") as cert_file:
                tls_cert_chain = cert_file.read()
            with open(self.private_key, "rb") as key_file:
                tls_private_key = key_file.read()
            self.connection_args["cert_chain"] = tls_cert_chain
            self.connection_args["private_key"] = tls_private_key

        self.location = f"{self.scheme}://{self.host}:{self.port}"

        self.client = pyarrow.flight.FlightClient(self.location, **self.connection_args)

        loginfo(f" | CONNECTING TO ARROW FLIGHT SERVER ... {self.location} ...")

        while True:
            try:
                action = pyarrow.flight.Action("healthcheck", b"")
                options = pyarrow.flight.FlightCallOptions(timeout=1)
                list(self.client.do_action(action, options=options))
                break
            except pyarrow.ArrowIOError as e:
                logerror(" | CANT CONNECTED TO ARROW SERVER !")
                if "Deadline" in str(e):
                    loginfo(" | SERVER IS NOT READY. WHAITING ...")

        loginfo(f" | -> CONNECTED !")

    def list_flights(args, client, connection_args={}):
        loginfo(f" | FLIGHTS -> ")
        for flight in client.list_flights():
            descriptor = flight.descriptor
            if descriptor.descriptor_type == pyarrow.flight.DescriptorType.PATH:
                loginfo(f" | - PATH: {descriptor.path}")
            elif descriptor.descriptor_type == pyarrow.flight.DescriptorType.CMD:
                loginfo("Command:", descriptor.command)
            else:
                loginfo("Unknown descriptor type")

            loginfo("Total records:", end=" ")
            if flight.total_records >= 0:
                loginfo(flight.total_records)
            else:
                loginfo("Unknown")

            loginfo("Total bytes:", end=" ")
            if flight.total_bytes >= 0:
                loginfo(flight.total_bytes)
            else:
                loginfo("Unknown")

            loginfo("Number of endpoints:", len(flight.endpoints))
            loginfo("Schema:")
            loginfo(flight.schema)
            loginfo("---")

        loginfo("\nActions\n=======")
        for action in client.list_actions():
            loginfo("Type:", action.type)
            loginfo("Description:", action.description)
            loginfo("---")

    def do_action(args, client, connection_args={}):
        try:
            buf = pyarrow.allocate_buffer(0)
            action = pyarrow.flight.Action(args.action_type, buf)
            loginfo("Running action", args.action_type)
            for result in client.do_action(action):
                loginfo("Got result", result.body.to_pybytes())
        except pyarrow.lib.ArrowIOError as e:
            loginfo("Error calling action:", e)

    def read_data(self, path=None, command=None, connection_args={}):
        if path:
            descriptor = pyarrow.flight.FlightDescriptor.for_path(*path)
        else:
            descriptor = pyarrow.flight.FlightDescriptor.for_command(command)

        info = self.client.get_flight_info(descriptor)
        for endpoint in info.endpoints:
            loginfo(f" | TICKET: {endpoint.ticket}")
            for location in endpoint.locations:
                location = self.location
                loginfo(f" | LOCATION: {location}")
                get_client = pyarrow.flight.FlightClient(location, **connection_args)
                reader = get_client.do_get(endpoint.ticket)
                self.data = reader.read_pandas()[0][0]

        return self.data

    def send_data(self, data, path=None, command=None, connection_args={}):
        if path:
            descriptor = pyarrow.flight.FlightDescriptor.for_path(*path)
        else:
            descriptor = pyarrow.flight.FlightDescriptor.for_command(command)
        loginfo(f" | SENDING DATA -> AFS | DESCRIPTOR: {descriptor}")
        loginfo(f" | SENDING DATA -> AFS | CREATE ARROW TABLE... ")
        data = pd.Series([data], dtype="string")
        df_data = pd.DataFrame(data)
        arrow_table = pyarrow.Table.from_pandas(df_data)
        loginfo(f" | SENDING DATA -> AFS | CREATE ARROW TABLE |  DONE ")
        loginfo(f" | SENDING DATA -> AFS ... ")
        writer, _ = self.client.do_put(descriptor, arrow_table.schema)
        writer.write_table(arrow_table)
        loginfo(f" | SENDING DATA -> AFS | DONE ")
        writer.close()
        loginfo(f" | AFS | CLOSE CONNECTION ")

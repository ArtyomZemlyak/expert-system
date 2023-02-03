import os
import json
import pathlib

from psycopg import connect, Error


class DBESPostgresAdapter:
    def __init__(self) -> None:
        self.client = None
        ##############################################
        #### CONFIG ##################################
        ##############################################
        self.__postgres_auth = {}
        path = pathlib.Path(__file__).parent.parent.parent
        path = os.path.join(path, r"config.json")
        with open(path) as json_file:
            self.__postgres_auth = json.load(json_file)["bi_postgres_auth"]
        ##############################################
        ##############################################

    def __del__(self) -> None:
        self.close()

    def connect(self):
        try:
            # Declare a new PostgreSQL connection object
            self.client = connect(
                dbname=self.__postgres_auth["db_name"],
                user=self.__postgres_auth["user"],
                host=self.__postgres_auth["host"],
                port=self.__postgres_auth["port"],
                password=self.__postgres_auth["pass"],
                connect_timeout=3,
            )
        except Exception as e:
            print(f" | CANT CONNECT TO POSTGRES DATABASE!")
            print(f" | {e}")

    def get(self, command_sql: str):
        try:
            cursor = self.client.cursor()
            cursor.execute(command_sql)
            result = list(cursor)
            if result is not None and result != "":
                try:
                    cursor.close()
                except Exception as e:
                    print(f" | CANT CLOSE CURSOR!")
                    print(f" | {e}")

            return result

        except (Exception, Error) as e:
            print(f" | psycopg CONNECT ERROR:")
            print(f" | {e}")

    def close(self):
        try:
            if self.client:
                self.client.close()
                print(f" | DISCONECTED FROM POSTGRES DATABASE")
        except (Exception, Error) as e:
            print(f" | psycopg DISCONNECT ERROR:")
            print(f" | {e}")

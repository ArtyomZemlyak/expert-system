#!/usr/bin/env python

import os

import ArrowFlightServer as afs


if __name__ == "__main__":

    afs_hendler = afs.ArrowFlightServer()
    afs_hendler.serve_start()

#!/usr/bin/env python

import uvicorn

if __name__ == "__main__":

    uvicorn.run(
        "FastAPI:app", host="0.0.0.0", port=80
    )  # reload - reload page after change file

#!/bin/bash
HIST_DATA=`pwd`/historical-data
docker run -it -v $HIST_DATA:/historical-data lc-tools

#!/bin/bash

THISDIR=`dirname $0`
ROOTDIR=`realpath  "$THISDIR"`

coverage run -m pytest --junitxml=pytest-report.xml  ./tests/

rm -fr "${ROOTDIR}/htmlcov/"

coverage html
coverage xml
coverage report
coverage json

echo browse "${ROOTDIR}/htmlcov/index.html"

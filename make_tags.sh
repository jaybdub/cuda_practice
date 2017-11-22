#!/bin/bash

source_files=main.cu

ctags --langmap=c++:+.cu $source_files

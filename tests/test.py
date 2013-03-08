#!/usr/bin/python2.7

import sys
import os


def usage_check():
  if len(sys.argv) != 3:
    usage_print()
    sys.exit(1)

def usage_print():
  print("usage: ./test.py <cpu,gpu> <stuff>")

def main():
  usage_check()

if __name__ == "__main__":
  main()

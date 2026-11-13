import sys


def main(argv):
    command = argv[1] if len(argv) > 1 else "serve"
    if command == "serve":
        from clef.serve import main as serve

        serve()
        return
    if command == "quantize":
        from clef.quantize import main as quantize

        quantize()
        return
    raise SystemExit("команда: serve или quantize")


if __name__ == "__main__":
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        sys.exit(130)

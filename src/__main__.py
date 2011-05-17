import importlib
from sys import argv


def main():
    if len(argv) < 2:
        print("no number given")
        exit(-1);
    hw = 0
    try:
        hw = int(argv[1])
    except:
        print('arg was not a number!')
        exit(-1)

    importlib.import_module(f'src.t{hw}')

    pass
if __name__ == "__main__":
    main()

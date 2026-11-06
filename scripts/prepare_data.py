import sys
from camf_recnas.cli import main

if __name__ == "__main__":
    sys.argv = ["camf-recnas", "prepare", *sys.argv[1:]]
    main()

"""

"""

from argparse import ArgumentParser
from pathlib import Path


def encapsulate_string(s: str) -> str:
    r = ""
    r += '\\'         # add backlash
    r += '"'          # add quote
    r += s            # add key
    r += '\\'         # add backlash
    r += '"'          # add quote
    return r


def isdigit(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def main() -> str:

    parser = ArgumentParser()
    parser.add_argument("file", type=Path, help="Path to the bash script.")
    args = parser.parse_args()
    file = args.file
    del args

    args = []
    add = False
    with open(file, "r") as fp:
        for line in fp:
            if add:
                args.append(line)
            if line.startswith("python"):
                add = True

    args = [a for a in args if not a.startswith("#")]
    args = [a.replace('"', "").replace("'", "").replace("\\", "").rstrip() for a in args]
    args = [f'"{a}"' for a in args]

    idx_1 = None
    idx_2 = None
    for i, a in enumerate(args):
        if "--arch_config" in a:
            if idx_1 is not None:
                raise RuntimeError()
            idx_1 = i
        if "--pretraining_checkpoint" in a:
            if idx_2 is not None:
                raise RuntimeError()
            idx_2 = i

    if idx_1 is not None:
        s = args[idx_1]
        s = s[len('"--arch_config={'):-len('}"')]
        iterator = [x.split(":") for x in s.split(",")]
        s = ""
        for k, v in iterator:
            k = k.strip()
            v = v.strip()
            s += encapsulate_string(k)
            s += ": "

            if v in ("true", "false", "null") or isdigit(v):
                s += v
            else:
                s += encapsulate_string(v)

            s += ", "

        s = s[:-len(", ")]
        s = '"--arch_config={' + s + '}"'
        args[idx_1] = s

    if idx_2 is not None and all(c in args[idx_2] for c in ("{", "}")):
        s = args[idx_2]
        s = s[len('"--pretraining_checkpoint={'):-len('}"')]
        iterator = [x.split(":") for x in s.split(",")]
        s = ""
        for k, v in iterator:
            k = k.strip()
            v = v.strip()
            s += encapsulate_string(k)
            s += ": "

            if v in ("true", "false", "null") or isdigit(v):
                s += v
            else:
                s += encapsulate_string(v)

            s += ", "

        s = s[:-len(", ")]
        s = '"--pretraining_checkpoint={' + s + '}"'
        args[idx_2] = s

    args_ = []
    for a in args:
        parts = a.split(" ")
        if len(parts) == 0:
            continue
        if len(parts) == 1:
            args_.append(a)
            continue
        a = parts[0]
        b = " ".join(parts[1:])
        args_.append(f"{a}={b}")
    args = args_
    del args_

    print(", ".join(args))


if __name__ == "__main__":
    main()

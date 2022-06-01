def main():
    with open("README.md", "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open("README.md", "w", encoding="utf-8") as f:
        in_style_tag = False
        for line in lines:
            if "?it/s]" in line:
                continue
            if "<style scoped>" in line:
                in_style_tag = True
                continue
            if "</style>" in line:
                in_style_tag = False
                continue
            if in_style_tag:
                continue

            f.write(line)


if __name__ == "__main__":
    main()

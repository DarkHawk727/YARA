import os

class PathChecker:
    def __init__(self):
        self.paths = []

    def check_file_or_directory(self, path):
        if os.path.isfile(path):
            self.paths.append(path)
            return f"{path} is a file."
        elif os.path.isdir(path):
            return f"{path} is a directory."
        else:
            return f"{path} is neither a file nor a directory."

    def list_files_in_directory(self, directory):
        if os.path.isdir(directory):
            self.paths.append(directory)
            files = os.listdir(directory)
            if files:
                print(f"Files in {directory}:")
                for file in files:
                    file_path = os.path.join(directory, file)
                    self.paths.append(file_path)
                    print(file_path)
            else:
                print(f"{directory} is an empty directory.")
        else:
            print(f"{directory} is not a directory.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check if a path is a file or directory.")
    parser.add_argument("paths", nargs="+", help="List of paths to check")

    args = parser.parse_args()

    path_checker = PathChecker()
    
    for path in args.paths:
        result = path_checker.check_file_or_directory(path)
        print(result)

        if os.path.isdir(path):
            path_checker.list_files_in_directory(path)

    print("Stored paths:")
    for stored_path in path_checker.paths:
        print(stored_path)

from pathlib import Path
from shutil import copytree, ignore_patterns
from argparse import ArgumentParser

parse = ArgumentParser('python3 mk_pjt.py {new loc} [-n | --name] [pjt name]')
parse.add_argument('-n', '--name', type=str, default='new_project')
parse.add_argument('path', type=Path, default='../')
args = parse.parse_args()

target_dir = args.path
assert target_dir.is_dir(), '주어진 매개변수는 파일이 아닙니다.'

project_name = args.name
target_dir = target_dir / project_name

current_dir = Path()
ignore = [".git", "saved", "mk_pjt.py", "README.md", "__pycache__"]
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
print('New project initialized at', target_dir.absolute().resolve())
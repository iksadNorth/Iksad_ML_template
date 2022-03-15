# %%
import re

from pathlib import Path
from glob import glob

# %%
class PathTree():
    MARKER = '$'
    REGEXR = f'\{MARKER}\w*'
    # 무조건 $~~~ 중 ~~~에 해당하는 변수는 리스트의 형태를 가지고 있어야 한다.
    def __init__(self, path,) -> None:
        self.path = path
        print(self.analyse())
    
    def update(self, dictionary):
        self.__dict__.update(dictionary)
    
    # '$~~~' 찾아내기.
    def analyse(self):
        params = []
        
        list_path = glob(f"{self.path}/**", recursive=True)
        regexr = re.compile(self.REGEXR)
        for path in list_path:
            match = regexr.findall(path)
            params.extend(match)
        return set(params)
    
    def replace(self, new):
        # 복사하려는 경로 new 아래의 모든 디렉토리를 재귀적으로 수집.
        list_path = glob(f"{self.path}/**", recursive=True)

        # $ 표시가 사라질 때까지 반복하면서 해당 내용 채우기.
        while any([(self.MARKER in path) for path in list_path]):
            for idx, path in enumerate(list_path):
                # '$~~~'를 찾기.
                match = re.search(self.REGEXR, path)
                
                # 있으면 대체하고 없으면 그대로 두기
                if match:
                    string = list_path.pop(idx)
                    
                    query = match.group()
                    query_without_symbol = query.replace(self.MARKER, '', 1)
                    
                    # 해당 변수가 있으면 대체하고 
                    if hasattr(self, query_without_symbol):
                        list_parameter = getattr(self, query_without_symbol)
                        assert '__iter__' in list_parameter.__dir__(), f"{query_without_symbol}는 무조건 Iterable한 형태여야 함."
                        
                        for para in list_parameter:
                            new_path = string.replace(query, para, 1)
                            list_path.append(new_path)
                    # 해당 변수가 없으면 '$~~~'를 '~~~'로 바꿈.
                    else:
                        new_path = string.replace(query, query_without_symbol, 1)
                        list_path.append(new_path)

        # 모든 경로를 상대경로로 바꾸고 새로운 경로 new에 가져다 붙임.
        list_path = [Path(path).relative_to(Path(self.path).parent) for path in list_path]
        list_path = [new / path for path in list_path]
        
        return list_path
    
    def mktree(self, new_path):
        list_path = self.replace(new_path)
        
        # 해당 경로들 생성.
        for path in list_path:
            # 폴더 여부.
            if path.stem == path.name:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    
    def find(self, path_parent, path_piece):
        list_path = glob(f"{path_parent}/**", recursive=True)
        
        for path in list_path:
            if Path(path).stem==path_piece:
                return path
    
# %%
if __name__ == '__main__':
    k = PathTree('/opt/ml/workspace/template/saved/$project_name')
    
    k.project_name = ['pjt1', 'pjt2']
    k.SubFolder = ['Artifact', 'Config', 'Metrics', 'Model']
    k.text = ['README', 'TEST']
    
    k.mktree('/opt/ml/workspace/template/saved/test')
    
    print(k.find('/opt/ml/workspace/template/saved/test', 'Config'))
    
# %%

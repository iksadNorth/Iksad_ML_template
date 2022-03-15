# %%
import re

from pathlib import Path
from glob import glob

# %%
class PathTree():
    """해당 모듈은 프로젝트 디렉토리의 유연성을 위해 고안됨.
    폴더 앞에 MARKER 즉, '${변수명}'와 같은 형식으로 이름을 
    주면 해당 클래스의 내부 필드를 이용해 자동으로 폴더 및 파일을 채워준다.
    자세한 예시는 맨 아래에 표현되어 있으므로 참고하면 이해가 빠를 것이다.

    Returns:
        Console: 해당 클래스의 인스턴스를 만들 때 마다 $가 앞에 붙어 있던 변수들의 목록을 출력한다.
    """
    MARKER = '$'
    REGEXR = f'\{MARKER}\w*'
    
    
    # 무조건 $~~~ 중 ~~~에 해당하는 변수는 리스트의 형태를 가지고 있어야 한다.
    # Ex) 폴더명 $Pjt
    #   ) self.Pjt = ['CoAtNet_Experiment']
    def __init__(self, path,) -> None:
        self.path = path
        print(self.analyse())
    
    def update(self, dictionary):
        self.__dict__.update(dictionary)
    
    # $가 앞에 붙어 있던 변수들의 목록.
    def analyse(self):
        params = []
        
        list_path = glob(f"{self.path}/**", recursive=True)
        regexr = re.compile(self.REGEXR)
        for path in list_path:
            match = regexr.findall(path)
            params.extend(match)
        return set(params)
    
    def _replace(self, new):
        """self.path에 저장된 경로아래의 모든 디렉토리를 감지하고
        ${폴더} or ${파일}.txt 형식의 폴더, 파일을 변형해 채워 넣는다.
        단, 실제로 물리적으로 생성한다기 보다 만들 경로들을 List형태로 출력한다.

        Args:
            new (pathlib.Path or str): 실제로 만들 경로의 root 경로.

        Returns:
            List[pathlib.Path]: ${폴더} or ${파일}.txt 형식의 폴더, 파일을 변형한 경로 List
        """
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
        """실제로 해당 경로들을 폴더에 만들어 줄 메서드.

        Args:
            new_path (pathlib.Path or str): 실제로 만들 경로의 root 경로.
        """
        list_path = self._replace(new_path)
        
        # 해당 경로들 생성.
        for path in list_path:
            # 폴더 여부.
            if path.stem == path.name:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    
    def find(self, path_parent, path_piece):
        """해당 모듈을 사용하면 디렉토리 구조가 수시로 바뀌기 때문에
        필요한 디렉토리를 찾아주는 메소드가 요구됨.

        Args:
            path_parent (str): 찾고 싶은 경로의 root
            path_piece (str)): 찾고 싶은 경로의 이름.

        Returns:
            str: 찾고 싶은 경로의 절대 경로.
        """
        list_path = glob(f"{path_parent}/**", recursive=True)
        
        for path in list_path:
            if Path(path).stem==path_piece:
                return path
    
# %%
if __name__ == '__main__':
    '''
    현재 디렉토리 구조
    - saved
        - $project_name
            - $SubFolder
    '''
    k = PathTree('/opt/ml/workspace/template/saved/$project_name')
    # [console]
    # {'$project_name', '$SubFolder'}
    
    k.project_name = ['pjt1', 'pjt2']
    k.SubFolder = ['Artifact', 'Config', 'Metrics', 'Model']
    k.text = ['README', 'TEST']
    
    k.mktree('/opt/ml/workspace/template/saved/test')
    '''
    변형 후 디렉토리 구조
    - test
        - pjt1
            - Artifact
            - Config
            - Metrics
            - Model
        - pjt2
            - Artifact
            - Config
            - Metrics
            - Model
    '''
    
    print(k.find('/opt/ml/workspace/template/saved/test', 'Config'))
    # [console]
    # /opt/ml/workspace/template/saved/test/pjt1/Config
    
# %%

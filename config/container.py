# %%
import json
from copy import deepcopy

from argparse import ArgumentParser
from turtle import update

from matplotlib import container

# %%
class Container():
    """해당 클래스는 ArgumentParser에서 입력받은 환경 변수와, Json 파일로부터 입력받은 환경 변수를 
    일률적으로 관리하기 위해 설계된 클래스다. 
    변수를 보관하는 보관소는 총 3개가 있다. 
    
    1. ArgumentParser       [실행마다의 유연성 확보 but 사용할 때 번거로움.]
    2. Json 파일            [비교적 사용하기 쉽고 가독성이 좋음 but 유연성이 떨어짐.]
    3. Container 내부 필드  [가장 사용하기 쉽고 유연성 좋음 but 가독성이 떨어지고 휘발성을 가짐.]
    (이하 보관소 1, 보관소 2, 보관소 3 라고 칭함.)
    
    보관소 1과 보관소 2의 내용 훼손을 지양하여 설계했고 
    최종 결과물의 종합은 보관소 3에서 이뤄지도록 설계.
    
    결과물의 종합은 ._setup() 메서드를 이용해 실현.
    
    또 결과물을 Json 파일 형태로 저장할 수 있게 .record() 메서드로 실현.
    """
    _IGNORE = ['json', 'argparse']
    def __init__(self, argparse:ArgumentParser=None, json_path:str=None):
        self.argparse = None
        self.json = {}
        
        if json_path:
            self._open_json(json_path)
        if argparse:
            self.argparse = argparse
        
        self._setup()
        
    @classmethod
    def inherit(cls, dictionary:dict):
        obj = cls()
        obj.__dict__.update(dictionary)
        return obj
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    ######################################################
    # Json 
    def _open_json(self, path):
        with open(path, 'r') as f:
            new_dict = json.load(f)
            self.json = new_dict
    
    def _update_json(self, path):
        with open(path, 'r') as f:
            new_dict = json.loads(f)
            self.json.update(new_dict)
        
        self._setup()
    
    def record(self, path:str):
        with open(path, 'w') as f:
            stuff = self.json
            json.dump(stuff, f)
                
    ######################################################
    # 정보 종합
    def _setup(self):
        if self.json:
            new_dict_json = dict(self.json)
            self.__dict__.update(new_dict_json)
            
        if self.argparse:
            new_dict_argparse = self.argparse.__dict__
            self.__dict__.update(new_dict_argparse)
    
    ######################################################
    # 출력    
    def dict(self):
        # 쓸모없는 필드 제거 후 출력.
        stuff = {k:v for k, v in self.__dict__.items() if not k in self._IGNORE}
        return stuff
    
    def __repr__(self):
        BOUNDRY = '=' * 50
        GAP_KEY_VALUE = 3 * '\t'
        stuff = self.dict()
        
        result = f'''
In This Containers
{BOUNDRY}
Key\t{GAP_KEY_VALUE}Value
{BOUNDRY}
'''
        for k, v in stuff.items():
            result += f'{k:<10}{GAP_KEY_VALUE}{v}\n'
        
        return result
    
    ######################################################
    # 파라미터 전달 & 실제 모듈 로드
    def get_obj(self, obj, repo, default_repo=None):
        if hasattr(repo, obj):
            return getattr(repo, obj)
        print(f"Warning : There is no {obj} in {repo} ...")
        return getattr(default_repo, obj)

    def get_obj_with_param(self, param:str, local_repo, remote_repo=None, **kwargs_params):
        method = self.get_obj(self[param]['type'], local_repo, remote_repo)
        kwargs = Container.inherit(self[param]['args']).dict()
        kwargs.update(kwargs_params)
        return method(**kwargs)

# %%
if __name__ == '__main__':
    a = Container()
    a.name = 'jung hoon'
    a.device = 's21'
    
    print(a)
    # [console]
    # In This Containers
    # ==============================
    # Key				Value
    # ==============================
    # name      			jung hoon
    # device    			s21
    

# %%
if __name__ == '__main__':
    b = Container(json_path='test.json')
    b.nick = 'iksadNorth'
    b.device = 'Lenovo'
    
    print(b)
    # In This Containers
    # ==============================
    # Key				Value
    # ==============================
    # name      			jung hoon
    # device    			Lenovo
    # nick      			iksadNorth
# %%

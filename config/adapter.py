# %%
from numpy import isin
from config.container import Container
from functools import partial

# %%
def adapter(func):
    """Container 객체만 파라미터를 사용해서 기존 메서드를 사용할 수 있게 하는 데코레이터.

    Args:
        func (Callable): 기존의 매개변수를 모두 Container 객체에서 사용할 수 있도록 함.
                         이 방법을 통해 이 프로젝트 만을 위해 번거롭게 파라미터를 변형치 않아도 됨.
    """
    def wrapper(*args, **kwargs):
        # container가 None 값이면 그냥 통과시키기.
        flag_pass = True
        args = list(args)
        # 인자 중 Container 객체가 있는지 확인.
        for arg in args:
            if isinstance(arg, Container):
                container, flag_pass = arg, False
                args.remove(arg)
                break
        # 인자 중 Container 객체가 없으면 걍 통과.
        if flag_pass:return func(*args, **kwargs) 
        
        # param_func : 대상 메서드(함수 func)의 입력 파라미터가 문자열로 존재하는 리스트.
        code = func.__code__
        num_args = code.co_argcount
        param_func = code.co_varnames[:num_args]
        
        # param_func에 속하는 Container의 attribute들을 골라냄.
        addition = {k:v for k, v in container.dict().items() if k in param_func}
        kwargs.update(addition)
        return func(*args, **kwargs)
    
    return wrapper

def adapter_class(func):
    """Container 객체만 파라미터를 사용해서 기존 메서드를 사용할 수 있게 하는 데코레이터.

    Args:
        func (Callable): 기존의 매개변수를 모두 Container 객체에서 사용할 수 있도록 함.
                         이 방법을 통해 이 프로젝트 만을 위해 번거롭게 파라미터를 변형치 않아도 됨.
    """
    def wrapper(self, container:Container, *args, **kwargs):
        # container가 None 값이면 그냥 통과시키기.
        # if not isinstance(container, Container):
        #     return func(*args, **kwargs)
        
        # param_func : 대상 메서드(함수 func)의 입력 파라미터가 문자열로 존재하는 리스트.
        code = func.__code__
        num_args = code.co_argcount
        param_func = code.co_varnames[:num_args]
        
        # param_func에 속하는 Container의 attribute들을 골라냄.
        addition = {k:v for k, v in container.dict().items() if k in param_func}
        # new_func = partial(func, **addition)
        # return new_func(self, *args, **kwargs)
        kwargs.update(addition)
        return func(self, *args, **kwargs)
    
    return wrapper

# %%
@adapter
def add(a:int,b:float, k=4):
    print(k)
    return a + b

# %%
if __name__ == '__main__':
    parser = Container()
    parser.a = 7
    parser.b = 6
    parser.incorrect_key = 'test_test'
    
    res = add(parser, k='test')
    print(res)
        
# %%
if __name__ == '__main__':
    res = add(None, 7, 6, k='test')
    print(res)

# %%
if __name__ == '__main__':
    res = add(1, parser, 6, k='test')
    print(res)
# %%

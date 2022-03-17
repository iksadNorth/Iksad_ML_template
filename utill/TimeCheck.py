# %%
from time import time
from time import sleep

# %%
class TimeCheck():
    FN_PRINT = print
    def __init__(self) -> None:
        self.queue = []
    
    def mark(self, name_flag:str):
        entity = name_flag, time()
        self.queue.append(entity)
        
    def mark_(self, name_flag:str):
        self.mark(name_flag)
        self.FN_PRINT(name_flag)
    
    def cal(self):
        result = {}
        for i, entity in enumerate(self.queue):
            if i == 0:
                previous_entity = entity
                continue
            else:
                name = f"[{previous_entity[0]}] --> [{entity[0]}]"
                diff = entity[1] - previous_entity[1]
                result[name] = diff
                previous_entity = entity
        
        return result
    
    def print(self, name_flag:str, fn_print=None):
        self.mark(name_flag)
        if not fn_print:
            fn_print = lambda k, v: self.FN_PRINT(f'{k:<50} \t{v:4.2f}초')
        
        self.FN_PRINT('=' * 50)
        for k, v in self.cal().items():
            fn_print(k, v)
        self.FN_PRINT('=' * 50)

        self.queue = []

# %%
if __name__ == '__main__':
    check = TimeCheck()
    
    check.mark('first')
    sleep(1)
    check.mark('second')
    sleep(1)
    check.mark('third')
    sleep(1)
    check.mark('fourth')
    sleep(1)
    
    check.print()
    
    
# %%

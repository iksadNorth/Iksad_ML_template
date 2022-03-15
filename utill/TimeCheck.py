# %%
from time import time
from time import sleep

# %%
class TimeCheck():
    def __init__(self) -> None:
        self.queue = []
    
    def mark(self, name_flag:str):
        entity = name_flag, time()
        self.queue.append(entity)
    
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
    
    def print(self, fn_print=None):
        if not fn_print:
            fn_print = lambda k, v: print(f'{k} {v:4.2f}')
        
        print('=' * 50)
        for k, v in self.cal().items():
            fn_print(k, v)
        print('=' * 50)

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

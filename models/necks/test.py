
#coding=utf-8
class Item:
    def info(self):
        print("Item中的方法",'这是一个商品')
        
class Product:
    def info(self):
        print('Product中的方法','这是一个能赚钱的商品')
        
class Computer(Item,Product):
    pass
    
c = Computer()
print(c)

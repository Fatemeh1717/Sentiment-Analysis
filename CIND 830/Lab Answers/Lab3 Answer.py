#!/usr/bin/env python
# coding: utf-8

# ### Lab 3 -Answer 

# ##### a)i.

# In[1]:


x=1/2
y= x<1/2
print(y)


# In[18]:


import math


# #### a)ii.

# In[15]:


x=4
y = (4 <= 4/(x-4))
print(y)   


# #### a)iii.

# In[21]:



from numpy.lib.scimath import logn
from math import e

x=1
y =logn(e, x) < (x-2)**4
print(y)


# #### a)iv.
# 

# In[23]:


x=2

z =x>=y
print(z)


# # b

# #### b)i

# In[24]:


x=3
z = 2<x<=4
print(z)


# #### b)ii

# In[31]:


x=2 
#o>=1
z = x<3 and o>=1
print(z)


# #### b)iii

# In[32]:


x= 2 
#c>=1
z= x<3 or c>= 1
print(z)


# #### b)iv

# In[ ]:


x =y=2
w=4
d = x<= y or y<w
print(d)


# # 2

# #### 2)a

# In[5]:



Number = float(input("Please input the number\n"))
if(Number>= 50):
   print("pass")
elif(Number <50):
   print("fail")
   



   


# #### 2)b

# In[10]:


Number = float(input("Please input the number\n"))
if(Number>= 50):
    print("pass")
else:
    print("fail")


# #### 2)c

# In[3]:


Number = float(input("Please input the number\n"))
if(Number>= 80 ):
    print("Excelent")
elif(Number>70):
    print("Good")
elif(Number >= 60):
    print("Satisfactory")
elif(Number >=50 ):
    print("Marginal")
else:
    print("Unsatisfactory")


# In[13]:


n1 = float(input("Please input the first positive number\n"))
n2 = float(input("Please input the second positive number\n"))

if (n1==n2):
    print("the shape would be Squre\n")
    area1 = n1*n2
    perimeter1= 4*n1
    print("Area of Square : ", area1)
    print("Perimeter of Square : ", perimeter1)
else:
    print("the shape would be Rectangle ")
    area2 = n1*n2
    perimeter2= 2*(n1+n2)
    print("Area of Rectangle : ", area2)
    print("Perimeter of Rectangle : ", perimeter2 )
    

    


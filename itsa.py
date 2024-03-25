# -*- coding: utf-8 -*-
"""ITSA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WVKUgpWU-rm21FSfRI5kjySxO96Yvcxz

2.問題描述：
撰寫一程式，可由鍵盤輸入英哩，程式的輸出為公里，其轉換公式如下：
1 英哩= 1.6 公里

輸入說明：
輸入欲轉換之英哩數(int)。

輸出說明：
輸出公里(double)，取到小數點以下第一位。

範例：

輸入範例:
90
95
輸出範例:
144.0
152.0
"""

A=float(input())
m=1.6*A
print(round(m,1))

"""3.問題描述：
有一圓形，直徑為200，且中心座標為(0,0)。請寫一支程式可以輸入「點」的座標，並判斷「點」是否在圓形的範圍內。如果「點」的位置剛好在邊界的話也算是在圓形範圍內(例：x=100，y=0)。

輸入說明：
輸入一整數座標，依序分別X與Y。

出說明：
輸出此座標位置在圓內或圓外訊息。
"""

x=float(input())
y=float(input())
if (x*x+y*y)**0.5<=100:
    print('inside')
else:
    print('outside')

"""4.問題描述：
假設某個停車場的費率是停車2小時以內，每半小時30元，超過2小時，但未滿4小時的部份，每半小時40元，超過4小時以上的部份，每半小時60元，未滿半小時部分不計費。如果您從早上10點23分停到下午3點20分，請撰寫程式計算共需繳交的停車費。

輸入說明：
輸入兩組時間，分別為開始與離開時間，24小時制。

輸出說明：
輸出停車費。
"""

time_in=input()
time_out=input()

m1=int(time_in[:2])*60+int(time_in[2:])
m2=int(time_out[:2])*60+int(time_out[2:])
gap=m2-m1

if gap<30:
  print('free')
elif gap>0 and gap<=120:
  pay=(gap//30)*30
elif gap>120 and gap<=240:
  pay=((gap-120)//30)*40+30*4
else:
  pay=((gap-240)//30)*60+40*4+30*4
print(pay)

"""5.問題描述：
撰寫一個程式，使用者輸入一個整數，印出8位元的二進制表示。

輸入說明：
輸入一個整數，介於-128～127之間。

輸出說明：
以8位元的二進制顯示。
"""

while True:
    try:
        num = int(input())
        if num<0:
            num = 256+num
        m = bin(num)
        o = m[2:]
        print(o.zfill(8))
    except:
        break

"""6.問題描述：
試撰寫一程式，可輸入月份，然後判斷其所屬的季節（ 3~5 月為春季，6~8 月為夏季， 9~11 月為秋季， 12~2 月為冬季）。

輸入說明：
輸入月份。

輸出說明：
輸出該月份的季節， 3~5 月為春季(Spring)， 6~8 月為夏季(Summer)， 9~11 月為秋季(Autumn)， 12~2 月為冬季(Winter)。
"""

month = int(input())
if month>=3 and month<=5:
  print('Spring')
elif month>=6 and month<=8:
  print('Summer')
elif month>=9 and month<=11:
  print('Autumn')
else:
  print('Winter')

"""7.問題描述 ：

在做傅立葉轉換時，常會用到複數，但每次都要分開來計算實部與虛部，非常的麻煩，現在透過operator overloading的方式來簡化程式設計師的負擔。須做加減乘。

輸入說明 ：

第一列輸入一個正整數n。其後有n列，每一列代表一個想要做運算的虛數，每一列之資料依序為運算元、虛數1、虛數2。虛數的格式為a b。

輸出說明 ：

每一列表一個運算結果。虛數的格式為a b
"""

while True:
    try:
        n = int(input())
        while(n>0):
            w,a1,a2,b1,b2 = map(str,input().split())
            a1 = int(a1)
            a2 = int(a2)
            b1 = int(b1)
            b2 = int(b2)
            if w == '+':print((a1+b1),(a2+b2))
            elif w == '-':print((a1-b1),(a2-b2))
            elif w == '*':print((a1*b1-a2*b2),(a2*b1+a1*b2))
            elif w == '/':print(((a1*b1+a2*b2)/(b1*b1+b2*b2),(a2*b1-a1*b2)/(b1*b1+b2*b2)))
            n= n-1
    except:break

"""8.問題描述：
試撰寫一個程式，由輸入一個整數，然後判別此數是否為質數。質數是指除了 1 和它本身之外，沒有其它的數可以整除它的數，例如， 2, 3, 5, 7 與 11 等皆為質數。

輸入說明：
輸入一個正整數。

輸出說明：
質數顯示 YES ；非質數顯示 NO 。
"""

N=int(input())
count=0
for i in range(2,N):
    if (N%i==0):
        count+=1
if (count==0):
    print('YES')
else:
    print('NO')

"""9.問題描述：
試寫一個程式，輸入一正整數N，可計算出1到N之間可被3整除的數值之總和。

輸入說明：
輸入一正整數。

輸出說明：
輸出總和。
"""

N=int(input())
num=0
for i in range(1,N+1):
    if i%3==0:
        num+=i
    else:
        continue
print(num)

"""10.問題描述：

給定二個正整數，利用輾轉相除法求其最大公因數。

輸入說明：

給定二個正整數

輸出說明：

輸出最大公因數

範例：

假設輸入為 300 與 250, 則輸出為 50
"""



"""11.問題描述 ：

請設計一程式，輸入一個陣列並且反轉後再輸出。



輸入說明 ：

第一行先輸入矩陣的行、列，之後再輸入陣列元素。



輸出說明 ：

反轉後的矩陣。
"""



"""12.問題描述：

給定下列遞迴函式 :

![C_RU06.JPG](data:image/jpeg;base64,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)

請計算出 f (k) 。

輸入說明：

輸入值為一個大於 1 的整數。

輸出說明：

f(k) 的計算結果。
"""

def f_k(n):
  if n==0 or n==1:
    return n+1
  elif n>1:
    return f_k(n-1)+f_k(n//2)
while True:
    try:
        n=int(input())
        print(f_k(n))
    except:
        break

"""13.問題描述 ：

撲克牌的遊戲有很多種，像是大老二、撿紅點等。然而，現在您要參與的是比大小遊戲，每張牌各有其花色和數字，大小比較主要以花色為主，黑桃 > 紅心 > 方塊 > 梅花；倘若花色相同時，則比較數字。

輸入說明 ：

第一列的整數，代表撲克牌的疊數，其後有若干列，每列即為一疊牌的內容，每張牌分別以英文、數字作表示，其中 S 代表黑桃、 H 代表紅心、 D 代表方塊、 C 代表梅花。每筆資料分別以空白隔開。

輸出說明 ：

印出排列過後的撲克牌。一行是一 疊 牌， 每張牌以空白隔開。

驗證:H5 D4 S2 C13

D8 S3 D10 C12 H7

H6 S3

C5 D11 S1

正確輸出:
S2 H5 D4 C13

S3 H7 D10 D8 C12

S3 H6

S1 D11 C5
"""

lst = input().split()
shape = {'S':4,'H':3,'D':2,'C':1} #符號 黑桃S 紅心H 方塊D 梅花C
#先比花色#利用排序觀念
for i in range(len(lst)):
  for j in range(i,len(lst)):
    if shape[lst[i][0]]<shape[lst[j][0]]:
      lst[i],lst[j]=lst[j],lst[i]
    elif shape[lst[i][0]]==shape[lst[j][0]]:
      if int(lst[i][1:])<int(lst[j][1:]):
        lst[i],lst[j]=lst[j],lst[i]
for i in range(len(lst)):
  print(lst[i],end=' ')

"""14.問題描述：
迴文是指從前面讀和從後面讀都相同的一個數字或一段文字。例如下列每一五位數的整數都是迴文： 123321 ， 55555 ， 45554 ， 11611 。請撰寫一個程式，判斷它是否迴文。

輸入說明：
輸入一個正整數。

輸出說明：
迴文印出 ” 是 ” ；非回文印出 ” 否 ” 。
"""

S = input()
if S[::1]==S[::-1]:
  print('是')
else:
  print('否')

"""15.問題描述 ：

在電腦科學上 ，計算一串文字上各個字母出現的頻率是常被用到的技術，這對壓縮來講是很重要的資訊，而計算字數也可以幫助人們作校正的工具。一行文字被空白、逗點或是句點所分隔而形成很多字，例如 ”I have a pencil.” 這行字就有 I ， have ， a ， pencil 這四個字，即此行字數為 4 。所以現在要請你幫忙設計一個程式來計算一行文字的字 數及各個字母出現的次數。

輸入說明 ：

輸入一行正常的英文文字，也就是不要有開頭是空白或是有連續兩個 空白的情形發生，並且內容只能包含英文字母、空白、逗點、句點。 注意 : 輸入的字串長度最多是 100 。

輸出說明 ：

第一行輸出一個正整數 n ，表示此行文字的字數。 第二行開始依序輸出在此行文字中有出現的字母及出現的次數。 注意 : 大小寫不分，輸出小寫字母。
"""

a = [str(j) for j in input().split()]
abc={}
for i in range(len(a)):
  a[i]=a[i].lower()
for i in range(len(a)):
  for j in range(len(a[i])):
    if a[i][j] not in abc:
      abc[a[i][j]]=1
    else:
      abc[a[i][j]]+=1
print(abc)

"""***16. 子字串出現次數***
問題描述：

給予兩個英文字串，請計算出第一個字串出現在第二個字串中的次數。

輸入說明：

輸入分為兩行，第一行是由英文大小寫字母與數字所組成的字串，長度不超過 128 個字母。

第二行也是由英文大小寫字母與數字所組成的字串，長度不超過 512 個字母。

輸出說明：

第一個字串出現在第二個字串中的次數。
"""

N = input()#字句
M = input()#字母
num=0
for i in range(len(N)):
  if N[i:i+len(M)] == M:
    num+=1
print(num)

"""***題目17. 英文斷詞***
問題描述 ：

斷詞在自然語言的研究上是個很重要的步驟，主要就是將關鍵字從句子中斷出，英文的斷詞較為簡單，就根據句子中的空格將英文字隔開。

輸入說明 ：

輸入一句英文敘述句。 字元數≤1000。

輸出說明 ：

將輸入的句子進行斷詞，將斷出的關鍵字依照句子中的出現排序列印出。全部轉成小寫，列印出的關鍵字不得重複，關鍵字間以一個空格隔開，最後一個關鍵字後面進行換行。例如輸入 How do you do ，則輸出 how do you 。
"""

lst = input().split()
for i in range(0,len(lst)):
  for j in range(i+1,len(lst)):
    if lst[i] == lst[j]:
      lst.pop(j)
for i in range(len(lst)):
  print(lst[i].lower(),end=' ')

lst_new=[]
lst=input().split()
for i in lst:
  if i.lower() not in lst_new:
    lst_new.append(i.lower())
for j in lst_new:
  print(j,end=' ')

"""***題目18. QWERTY***
Time Limit: 1 seconds

問題描述 ：

輸入一段文字 ( 限制為 ASCII 表中，編號 32 至 125 之字元 ) ，將它們每個字元依照鍵盤的位置，印出它們右邊的字元，若右邊按鍵有兩層字元 ( 如 : 和 ; 位於同一按鍵上 ) ，則輸出下層字元，即 ”;” ，若該按鍵為上層字元 ( 如 !) 則輸出其右邊按鍵之上層字元 ( 如 @) ，若輸入的字元右邊的鍵為不可視按鍵，如 shift, enter,backspace 或右邊已無按鍵，如不做更動，照樣輸出。

輸入說明 ：

輸入一行鍵盤上屬於 ASCII 表中編號 32 至 125 之字元，並以 enter 結束該行。

輸出說明 ：

輸出該行每個字元右邊位置的文字，除題目所述之例外字元除外。
最後必須有換行
"""



"""***題目19. 最少派車數***
問題描述 ：

某遊覽車派遣公司共收到n筆任務訂單，訂單中詳細記載發車時間s和返回時間d。每一輛遊覽車只要任務時間不衝突，可立即更換司機繼續上路執行任務。請問該公司至少需要調遣多少車輛才足以應付需求？

輸入說明 ：

程式的輸入包含兩行數字，第一行包含一個正整數n，1 ≤ n ≤ 30，代表第二行有n筆訂單的出發時間和返回時間s1, d1, s2, d2, ..., sn, dn，0 < si < di ≤ 24，而此2n個正整數間以空格隔開。

輸出說明 ：

輸出最少車輛需求數。
"""



"""***題目20. 大整數加法***
問題描述：

有時候我們有些很大的值，大到即使大型的計算機也無法幫我們作一些很基本的運算。請你寫一個程式來解決兩個大整數的加法問題。

輸入說明：

第一行有一個正整數 N ，表示共有 N 筆測試資料。接下來有 N 行，每行為一筆測試資料，內含兩個整數，其值不超過 30 位數，兩個整數間有一個空格。

輸出說明：

每筆測試資料輸出兩個整數的和於一行。
"""

while True:
  try:
    n = int(input())
    for i in range(n):
      n1,n2 = map(int,input().split())
      print(n1+n2)
  except:
    break

"""21.問題描述 ：

寫一個程式來找出輸入的十個數字的最大值和最小值，數值不限定為整數，且值可存放於 float 型態數值內。

輸入說明 ：

輸入十個數字，以空白間隔。

輸出說明 ：

輸出數列中的最大值與最小值，輸出時需附上小數點後兩位數字。
"""

lst = [float(j) for j in input().split()]
max = lst[0]
min = lst[9]
for i in range(len(lst)):
  if max<lst[i]:
    max = lst[i]
  elif min>lst[i]:
    min = lst[i]
print(max,min)

"""***26.*** 各位數和排序
問題描述 ：

輸入一整數N及N個整數，請依照十進位中各位數字和由小到大排序輸出。如果各位數字和相等則比較數值由小到大排列。例如： 9122的各位數字和為 9+1+2+2=14、3128 的各位數字和為 3+1+2+8=14而5112的各位數字和為 5+1+1+2=9。所以輸入 9122 3128 5112 需輸出 5112 3128 9122 ，這是因為 5112(9) < 3128(14) < 9122(14)，其中又因為 3128 與 9122 兩者的各位數字和都是 14，所以將 數值小的 3128 放前面。
"""

N = int(input())
a = [str(j) for j in input().split(' ')]
num = [0]*N
for i in range(N):
  for j in range(len(a[i])):
    num[i] += int(a[i][j])
for i in range(N-1):
  for j in range(i+1,N):
    if num[i]>num[j]:
      num[i],num[j]=num[j],num[i]
      a[i],a[j]=a[j],a[i]
    elif num[i]==num[j] and int(a[i])>int(a[j]):
      a[i],a[j]=a[j],a[i]
for i in range(N):
   print(a[i],end=" ")
print()

"""29.身分證"""

N=input()
dict = {'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'J':18,'K':19,'L':20,'M':21,'N':22,'P':23,'Q':24,'R':25,'S':26,'T':27,'U':28,'V':29,'X':30,'Y':31,'W':32,'Z':33,'I':34,'O':35}
ABC="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
if len(N)!=10 and N[0] not in ABC:
  print('Wrong!!')
else:
  p = str(dict[ABC[0]])
  x1 = p[0]
  x2 = p[1]
  num=0
  for i in range(7):
    num+=(8-i)*int(N[i+1])
  N8 = int(N[8])
  N9 = int(N[9])
  P = int(x1) + 9*int(x2) +num +N8+N9
if int(P)%10==0:
  print('correct!!')
else:
  print('Wrong!!')

"""***題目34. 標準體重計算***
問題描述：
已知男生標準體重＝(身高－80 )*0.7；女生標準體重＝(身高－70)*0.6；試寫一個程式可以計算男生女生的標準體重。

輸入說明：
每個測資檔案會有多組測資案例。輸入兩個數值，依序代表為身高及性別（1代表男性；2代表女性）。
"""

num = input().split(' ')
boy = 0
girl = 0
if len(num)!=2:
  print("wrong!!")
else:
  if int(num[1])==1:
    boy = (int(num[0])-80)*0.7
    print(round(boy,1))
  elif int(num[1])==2:
    girl = int(num[0]-70)*0.6
    print(round(girl))
  else:
    print('Wrong')
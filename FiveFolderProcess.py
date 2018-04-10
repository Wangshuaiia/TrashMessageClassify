f = open('FenciResult1.txt', 'r', encoding='utf-8')
f1 = open('file1.txt', 'w', encoding='utf-8')
f2 = open('file2.txt', 'w', encoding='utf-8')
f3 = open('file3.txt', 'w', encoding='utf-8')
f4 = open('file4.txt', 'w', encoding='utf-8')
f5 = open('file5.txt', 'w', encoding='utf-8')
lines = f.readlines()
i = 0
# for i in range(len(lines)):
for line in lines:
    i += 1
    if i <= 160000:
        f1.write(line.strip() + '\n')
    elif i > 160000 and i <= 320000:
        f2.write(line.strip() + '\n')
    elif i > 320000 and i <= 480000:
        f3.write(line.strip() + '\n')
    elif i > 480000:
        f4.write(line.strip() + '\n')
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f4 = open('file4.txt', 'a', encoding='utf-8')
f = open('FenciResult2.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    i += 1
    if i <= 640000:
        f4.write(line.strip() + '\n')
    elif i >640000 :
        f5.write(line.strip() + '\n')
f4.close()
f5.close()
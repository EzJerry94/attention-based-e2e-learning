import csv
import matplotlib.pyplot as plt

LENGTH = 1500


# 读取csv至字典
csvFile = open("./loss_log/loss_info_1.txt", "r")
csvFile1 = open("./loss_log/loss_info_2.txt", "r")
#reader = csv.reader(csvFile)

reader = [each for each in csv.reader(csvFile, delimiter=';')]
reader1 = [each for each in csv.reader(csvFile1, delimiter=';')]

batch = []
loss = []
for row in reader:
    batch.append(float(row[0]))
    loss.append(float(row[1]))
csvFile.close()

batch1 = []
loss1 = []
for row in reader1:
    batch1.append(float(row[0]))
    loss1.append(float(row[1]))
csvFile1.close()

print(batch)
print(loss)

fig = plt.figure()
plt.plot(batch[:LENGTH], loss[:LENGTH], 'b')
plt.plot(batch1[:LENGTH], loss1[:LENGTH], 'r')
plt.savefig('test.png')
plt.show()

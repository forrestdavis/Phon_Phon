data = open('test.data', 'r')
train_reduce = open('voiceless_test.data', 'w')

#sibilants = ['s', 'z', 'S', 'Z', 'tS', 'dZ']
sibilants = ['p', 't', 'k', 'f', 'T']

count = 0
for line in data:
    line = line.strip()
    if not line:
        continue
    line = line.split('\t\t')
    ipa = line[1].split()
    if ipa[len(ipa)-1] in sibilants:
        print ipa
        count += 1

        for x in range(len(line)):
            if x == len(line)-1:
                train_reduce.write(line[x]+'\n')
            else:
                train_reduce.write(line[x]+'\t\t')

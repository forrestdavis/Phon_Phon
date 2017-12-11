
train_data = open('train.data', 'r')
train_reduce = open('please.data', 'w')


sibilants = ['s', 'z', 'S', 'Z', 'tS', 'dZ']
voiceless_cons = ['f', 't', 'k', 'p', 'T']

check_sibilants = sibilants[:6]
check_voiceless = voiceless_cons[:2]
print check_voiceless

count = 0
for line in train_data:
    line = line.strip()
    if not line:
        continue
    line = line.split('\t\t')
    ipa = line[1]
    ipa = ipa.split()
    if ipa[len(ipa)-1] in voiceless_cons and ipa[len(ipa)-1] not in check_voiceless:
        print ipa
        count += 1
        continue

    for x in range(len(line)):
        if x == len(line)-1:
            train_reduce.write(line[x]+'\n')
        else:
            train_reduce.write(line[x]+'\t\t')

print count

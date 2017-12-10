
words = open('words.txt', 'r')

dist = {}
for x in range(1, 17):
    dist[x] = 0

for word in words:
    word = word.strip()
    if not word:
        continue 
    size = len(word)
    if size == 16:
        print word
    dist[size] += 1

print dist

count = 0.0
total = 0.0
for key in dist:
    count += key * dist[key]
    total += dist[key]

print count/total
print count

count = 0
for key in dist:
    if key > 11:
        break
    count += key*dist[key]

print count

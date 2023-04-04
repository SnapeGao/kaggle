n, m, k = map(int, input().split())
arr = list(map(int, input().split()))
max1 = max(arr[0],arr[1])
max2 = min(arr[0], arr[1])
for i in range(2,n) :
    if arr[i] > max1 :
        max2 = max1
        max1 = arr[i]
    elif arr[i] > max2 :
        max2 = arr[i]
print((n//(k+1)) * (k * max1 + max2) + n%(k+1) * max1,end="")
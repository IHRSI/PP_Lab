#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void calculateTotalKernel(float* itemPrices, int* purchaseMatrix, float* individualTotals, int numFriends, int numItems) {
    int friendIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (friendIdx < numFriends) {
        float sum = 0.0f;
        for (int i = 0; i < numItems; i++) {
            sum += itemPrices[i] * purchaseMatrix[friendIdx * numItems + i];
        }
        individualTotals[friendIdx] = sum;
    }
}

int main() {
    int numItems, numFriends;

    printf("Enter the number of items available in the mall: ");
    scanf("%d", &numItems);

    float* h_itemPrices = (float*)malloc(numItems * sizeof(float));
    char** itemNames = (char**)malloc(numItems * sizeof(char*));

    for (int i = 0; i < numItems; i++) {
        itemNames[i] = (char*)malloc(50 * sizeof(char));
        printf("Enter name for item %d: ", i + 1);
        scanf("%s", itemNames[i]);
        printf("Enter price for %s: ", itemNames[i]);
        scanf("%f", &h_itemPrices[i]);
    }

    printf("\n--- SHOPPING MALL MENU ---\n");
    for (int i = 0; i < numItems; i++) {
        printf("%d. %s - $%.2f\n", i + 1, itemNames[i], h_itemPrices[i]);
    }

    printf("\nEnter the number of friends: ");
    scanf("%d", &numFriends);

    int* h_purchaseMatrix = (int*)malloc(numFriends * numItems * sizeof(int));
    float* h_individualTotals = (float*)malloc(numFriends * sizeof(float));

    for (int f = 0; f < numFriends; f++) {
        printf("\nFriend %d, enter quantities for each item:\n", f + 1);
        for (int i = 0; i < numItems; i++) {
            printf("  Quantity of %s: ", itemNames[i]);
            scanf("%d", &h_purchaseMatrix[f * numItems + i]);
        }
    }

    float *d_itemPrices, *d_individualTotals;
    int *d_purchaseMatrix;

    cudaMalloc((void**)&d_itemPrices, numItems * sizeof(float));
    cudaMalloc((void**)&d_purchaseMatrix, numFriends * numItems * sizeof(int));
    cudaMalloc((void**)&d_individualTotals, numFriends * sizeof(float));

    cudaMemcpy(d_itemPrices, h_itemPrices, numItems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_purchaseMatrix, h_purchaseMatrix, numFriends * numItems * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numFriends + threadsPerBlock - 1) / threadsPerBlock;

    calculateTotalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_itemPrices, d_purchaseMatrix, d_individualTotals, numFriends, numItems);

    cudaMemcpy(h_individualTotals, d_individualTotals, numFriends * sizeof(float), cudaMemcpyDeviceToHost);

    float grandTotal = 0.0f;
    printf("\n--- FINAL BILL ---\n");
    for (int f = 0; f < numFriends; f++) {
        printf("Friend %d Total: $%.2f\n", f + 1, h_individualTotals[f]);
        grandTotal += h_individualTotals[f];
    }
    printf("------------------\n");
    printf("Grand Total: $%.2f\n", grandTotal);

    cudaFree(d_itemPrices);
    cudaFree(d_purchaseMatrix);
    cudaFree(d_individualTotals);
    
    for (int i = 0; i < numItems; i++) free(itemNames[i]);
    free(itemNames);
    free(h_itemPrices);
    free(h_purchaseMatrix);
    free(h_individualTotals);

    return 0;
}
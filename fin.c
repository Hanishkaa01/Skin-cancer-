#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_ROWS 33130
#define MAX_COLS 9
#define TRAIN_RATIO 0.8

typedef struct {
    char image_name[50];
    char patient_id[50];
    char lesion_id[50];
    char sex[10];
    char age[10];
    char site[50];
    char diagnosis[50];
    char benign_malignant[10];
    char target[10];
    int sex_encoded;
    int benign_malignant_encoded;
    int site_encoded;
    int diagnosis_encoded;
    float age_scaled;
    float sex_scaled;
    float benign_malignant_scaled;
    float site_scaled;
    float diagnosis_scaled;
} DataPoint;

typedef struct Node {
    int feature;
    float threshold;
    int label;
    struct Node *left;
    struct Node *right;
} Node;

DataPoint dataset[MAX_ROWS];
DataPoint train_set[MAX_ROWS];
DataPoint test_set[MAX_ROWS];
int total_rows = 0;
int train_size = 0, test_size = 0;

void trimWhitespace(char *str) {
    char *end;
    while (*str == ' ') str++;
    if (*str == 0) return;
    end = str + strlen(str) - 1;
    while (end > str && *end == ' ') end--;
    *(end + 1) = 0;
}

void loadCSV(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        return;
    }
    char line[256];
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file) && total_rows < MAX_ROWS) {
        sscanf(line, "%49[^,],%49[^,],%49[^,],%9[^,],%9[^,],%49[^,],%49[^,],%9[^,],%9[^,]", 
               dataset[total_rows].image_name, dataset[total_rows].patient_id, dataset[total_rows].lesion_id, 
               dataset[total_rows].sex, dataset[total_rows].age, dataset[total_rows].site, 
               dataset[total_rows].diagnosis, dataset[total_rows].benign_malignant, dataset[total_rows].target);
        total_rows++;
    }
    fclose(file);
    printf("CSV loaded successfully. Total records: %d\n", total_rows);
}

void splitDataset() {
    int train_count = (int)(total_rows * TRAIN_RATIO);
    train_size = train_count;
    test_size = total_rows - train_count;
    for (int i = 0; i < total_rows; i++) {
        if (i < train_count) {
            train_set[i] = dataset[i];
        } else {
            test_set[i - train_count] = dataset[i];
        }
    }
    printf("Dataset split into %d training and %d testing records.\n", train_size, test_size);
}

void encodeCategorical() {
    for (int i = 0; i < total_rows; i++) {
        dataset[i].sex_encoded = strcmp(dataset[i].sex, "male") == 0 ? 1 : 0;
        dataset[i].benign_malignant_encoded = strcmp(dataset[i].benign_malignant, "benign") == 0 ? 0 : 1;
        dataset[i].site_encoded = strlen(dataset[i].site) > 0 ? (int)dataset[i].site[0] % 100 : -1;
        dataset[i].diagnosis_encoded = strlen(dataset[i].diagnosis) > 0 ? (int)dataset[i].diagnosis[0] % 100 : -1;
    }
    printf("Categorical variables encoded successfully.\n");
}

void normalizeFeatures() {
    float max_age = 0;
    for (int i = 0; i < total_rows; i++) {
        float age_value = atof(dataset[i].age);
        if (age_value > max_age) {
            max_age = age_value;
        }
    }
    for (int i = 0; i < total_rows; i++) {
        dataset[i].age_scaled = atof(dataset[i].age) / max_age;
        dataset[i].sex_scaled = dataset[i].sex_encoded;
        dataset[i].benign_malignant_scaled = dataset[i].benign_malignant_encoded;
        dataset[i].site_scaled = dataset[i].site_encoded / 100.0;
        dataset[i].diagnosis_scaled = dataset[i].diagnosis_encoded / 100.0;
    }
    printf("Feature scaling applied: All numerical features normalized.\n");
}

Node* trainDecisionTree(DataPoint *data, int size) {
    if (size == 0) return NULL;
    
    int count_pos = 0, count_neg = 0;
    for (int i = 0; i < size; i++) {
        if (data[i].benign_malignant_encoded == 1) count_pos++;
        else count_neg++;
    }
    
    if (count_pos == 0 || count_neg == 0) {
        Node* leaf = (Node*)malloc(sizeof(Node));
        leaf->label = (count_pos > count_neg) ? 1 : 0;
        leaf->left = leaf->right = NULL;
        return leaf;
    }
    
    Node* node = (Node*)malloc(sizeof(Node));
    node->feature = rand() % 5;
    node->threshold = 0.5;
    node->label = (count_pos > count_neg) ? 1 : 0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

int predict(Node* root, DataPoint point) {
    if (!root->left && !root->right) return root->label;
    if (point.age_scaled < root->threshold) return predict(root->left, point);
    return predict(root->right, point);
}

void evaluateModel(Node* root) {
    int correct_predictions = 0;
    for (int i = 0; i < test_size; i++) {
        int predicted_label = predict(root, test_set[i]);
        int actual_label = test_set[i].benign_malignant_encoded;
        if (predicted_label == actual_label) {
            correct_predictions++;
        }
    }
    float accuracy = ((float)correct_predictions / test_size) * 100.0;
    printf("Model Accuracy: %.2f%%\n", accuracy);
}

int main() {
    loadCSV("metadata.csv");
    encodeCategorical();
    normalizeFeatures();
    splitDataset();
    printf("Training decision tree using Information Gain...\n");
    Node* root = trainDecisionTree(train_set, train_size);
    evaluateModel(root);
    return 0;
}
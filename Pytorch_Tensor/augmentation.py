import nltk
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


#IMAGE AUGUMENTATION

# Define a series of transformations
augmentations = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

# Load an image
image = Image.open(r'C:\Users\sittu\OneDrive\Desktop\AIML_Wipro\Pytorch_Tensor\dog.jpg')

# Apply transformations
augmented_image = augmentations(image)

# Convert the tensor to a PIL Image for visualization
augmented_image_pil = transforms.ToPILImage()(augmented_image)

# Display the original and augmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Augmented Image")
plt.imshow(augmented_image_pil)
plt.axis('off')

plt.show()




#TEXT AUGUMENTATION

#nltk.download('wordnet')            # Uncomment for 1st time
import random
from nltk.corpus import wordnet

# Synonym replacement
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

# Random insertion
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = wordnet.synsets(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0].lemmas()[0].name()
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

# Random swap
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random.randint(0, len(new_words)-1)
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

# Random deletion
def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    return new_words

# Example usage
sentence = "Python is a wonderful language."        # It has many features."
words = sentence.split()

print("Original Sentence:")
print(sentence)
print("\nSynonym Replacement:")
print(' '.join(synonym_replacement(words, 2)))
print("\nRandom Insertion:")
print(' '.join(random_insertion(words, 2)))
print("\nRandom Swap:")
print(' '.join(random_swap(words, 2)))
print("\nRandom Deletion:")
print(' '.join(random_deletion(words, 0.2)))

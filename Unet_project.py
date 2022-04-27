import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision.transforms import TenCrop, Resize, CenterCrop, GaussianBlur, RandomRotation, RandomEqualize
from torchvision.transforms.functional import rotate, equalize
import skimage.io
import random
from skimage.transform import rotate, AffineTransform, warp
from skimage.io import imread_collection
from skimage.util import random_noise
import elasticdeform.torch as etorch

datadir = 'EM_ISBI_Challenge_modified/'  # path to the directory containing data

###
### TODO: Changing U-net architecture, Further Augmentation traditional and new, Check original paper for inspiration
### TODO: Qualitative and quantitatice analysis of results
### TODO: Use baseline model to create psudo-prediciton labels for unlabeled test data, compare with our result
### TODO: Maybe compare our model output with true labels on a pixel basis, how far is the image vector from true vector


# Make dataset class.
class Data(torch.utils.data.Dataset):
    '''  Dataset which loads all images for training or testing'''

    def __init__(self, img_dir, label_dir, margin_size=20, augment=False):
        self.images = []
        self.labels = []
        self.load_pattern = os.path.join(os.getcwd(), img_dir, "*.png")
        self.images = imread_collection(self.load_pattern).concatenate() / 255
        self.load_pattern = os.path.join(os.getcwd(), label_dir, "*.png")
        self.labels = imread_collection(self.load_pattern).concatenate()
        self.labels = torch.from_numpy(np.double(self.labels[:, margin_size:-margin_size, margin_size:-margin_size] / 255))

        self.images.shape += (1,)
        self.images = torch.from_numpy(np.double(self.images)).permute(0, 3, 1, 2)
        # print(self.labels.shape)
        fig, ax = plt.subplots(2,2)
        fig.set_tight_layout(True)
        ax[0, 0].imshow(self.images[0, 0], cmap="gray")
        # ax[0, 0].imshow(self.labels[0])
        ax[0,0].axis('off')
        if augment:
            aug3 = torch.flip(self.images, [2])
            aug4 = torch.flip(self.images, [3])
            aug5 = torch.flip(self.images, [2, 3])
            lab3 = torch.flip(self.labels, [1])
            lab4 = torch.flip(self.labels, [2])
            lab5 = torch.flip(self.labels, [1, 2])
            aug_list = [aug3, aug4, aug5]
            lab_list = [lab3, lab4, lab5]


            # tencrop_im = Resize(512)(torch.cat(CenterCrop(512/2)(self.images)))
            # tencrop_lab = Resize(472)(torch.cat(CenterCrop(512/2)(self.labels)))
            # print(self.images.shape)
            # print(tencrop_im.shape)
            # print(tencrop_lab.shape)

            self.images = torch.cat((self.images, aug3, aug4, aug5))
            self.labels = torch.cat((self.labels, lab3, lab4, lab5))
            # self.images = torch.cat((self.images, tencrop_im))
            # self.labels = torch.cat((self.labels, tencrop_lab))
            new_images = []
            new_labels = []
            for im, lab in zip(self.images, self.labels):

                choice = random.randint(1,3)
                # rotate
                if choice == 1:
                    angle = random.randint(-180,180)
                    rotated_im = torch.tensor(rotate(im, angle=angle, mode='wrap'))
                    rotated_lab = torch.tensor(rotate(lab, angle=angle, mode='wrap'))
                    new_images.append(rotated_im)
                    new_labels.append(rotated_lab)

                # shift
                elif choice == 2:
                    a, b = random.randint(1, self.images.shape[-1]), random.randint(1, self.images.shape[-1])
                    transform = AffineTransform(translation=(a, b))
                    warped_im = torch.tensor(warp(im, transform, mode='wrap'))
                    warped_lab = torch.tensor(warp(lab, transform, mode='wrap'))
                    new_images.append(warped_im)
                    new_labels.append(warped_lab)


                # print(ok.shape)
                elif choice == 3:
                    displacement_val = np.random.randn(2, 3, 3) * 5
                    displacement = torch.tensor(displacement_val)
                    im_deformed = torch.unsqueeze(torch.tensor(etorch.deform_grid(im[0], displacement, order=3)), dim=0)
                    lab_deformed = torch.tensor(etorch.deform_grid(lab, displacement, order=3))
                    new_images.append(im_deformed)
                    new_labels.append(lab_deformed)
                #
                # new_images.extend([rotated_im, warped_im])#, im_deformed])
                # new_labels.extend([rotated_lab, warped_lab])#, lab_deformed])
                # new_images.extend([im_deformed])
                # new_labels.extend([lab_deformed])
                # print()

            # print(self.images.shape)
            # print(torch.stack(new_images).shape)
            self.images = torch.cat((self.images, torch.stack(new_images)))
            self.labels = torch.cat((self.labels, torch.stack(new_labels)))
            #
            # self.images = RandomEqualize(1)(self.images)


            ax[1, 0].imshow(aug3[0, 0], cmap="gray")
            # ax[1, 0].imshow(lab3[0])
            ax[1, 0].axis('off')
            ax[0, 1].imshow(aug4[0, 0], cmap="gray")
            # ax[0, 1].imshow(lab4[0])
            ax[0, 1].axis('off')
            ax[1, 1].imshow(aug5[0, 0], cmap="gray")
            # ax[1, 1].imshow(lab5[0])
            ax[1, 1].axis('off')
            plt.savefig('original_augmentation.png')

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)




TrainData = Data(datadir + 'train_images', datadir + 'train_labels', augment=True)
ValData = Data(datadir + 'validation_images', datadir + 'validation_labels')

#%% Check if implementation of dataset works as expected.

print(f'Nr. training images: {len(TrainData)}')
print(f'Nr. validation images: {len(ValData)}')


# %% Make model class.
class UNet128(torch.nn.Module):
    """Takes in patches of 128^2, returns 88^2"""

    def __init__(self, out_channels=2):
        super(UNet128, self).__init__()

        # Learnable
        self.conv1A = torch.nn.Conv2d(1, 8, 3)
        self.conv1B = torch.nn.Conv2d(8, 8, 3)
        self.conv2A = torch.nn.Conv2d(8, 16, 3)
        self.conv2B = torch.nn.Conv2d(16, 16, 3)
        self.conv3A = torch.nn.Conv2d(16, 32, 3)
        self.conv3B = torch.nn.Conv2d(32, 32, 3)
        self.conv4A = torch.nn.Conv2d(32, 16, 3)
        self.conv4B = torch.nn.Conv2d(16, 16, 3)
        self.conv5A = torch.nn.Conv2d(16, 8, 3)
        self.conv5B = torch.nn.Conv2d(8, 8, 3)
        self.convfinal = torch.nn.Conv2d(8, out_channels, 1)
        self.convtrans34 = torch.nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.convtrans45 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)

        # Convenience
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        # Down, keeping layer outputs we'll need later.
        l1 = self.dropout(self.relu(self.conv1B(self.dropout(self.relu(self.conv1A(x))))))
        l2 = self.dropout(self.relu(self.conv2B(self.dropout(self.relu(self.conv2A(self.pool(l1)))))))
        out = self.dropout(self.relu(self.conv3B(self.dropout(self.relu(self.conv3A(self.pool(l2)))))))

        # Up, now we overwritte out in each step.
        out = torch.cat([self.convtrans34(out), l2[:, :, 4:-4, 4:-4]], dim=1)
        out = self.dropout(self.relu(self.conv4B(self.dropout(self.relu(self.conv4A(out))))))
        out = torch.cat([self.convtrans45(out), l1[:, :, 16:-16, 16:-16]], dim=1)
        out = self.dropout(self.relu(self.conv5B(self.dropout(self.relu(self.conv5A(out))))))

        # Finishing
        out = self.convfinal(out)

        return out



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using cuda')
else:
    device = torch.device('cpu')
    print('Using cpu')



# Initiate the model, dataloaders and optimizer.

lr = 0.001
nr_epochs = 100


#  Loaders for training and testing set
trainloader = torch.utils.data.DataLoader(TrainData,
                                          batch_size=32,
                                          shuffle=True,
                                          drop_last=True)
valloader = torch.utils.data.DataLoader(ValData,
                                          batch_size=6)
model = UNet128().to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Pick anq image to show how predictitons change.
i = 50
im, lb = TrainData[i]
with torch.no_grad():
  lgt = model(im.unsqueeze(0).to(device, dtype=torch.float))
  prob = torch.nn.Softmax(dim=1)(lgt)

fig, ax = plt.subplots(1, 3, figsize=(10,5))
ax[0].imshow(im.permute(1,2,0))
ax[0].set_title('Image')
ax[1].imshow(lb)
ax[1].set_title('Label')
ax[2].imshow(prob[0,1].cpu().detach())
ax[2].set_title('Prediction')
plt.show()

epoch_losses = []
batch_losses = []
test_losses = []
iou_scores = []

# Train.
for epoch in range(nr_epochs):
    print(f'Epoch {epoch}/{nr_epochs}', end='')

    epoch_loss = 0.0
    for batch in trainloader:
        image_batch, label_batch = batch  # unpack the data
        image_batch = image_batch.to(device, dtype=torch.float)
        label_batch = label_batch.to(device, dtype=torch.long)

        logits_batch = model(image_batch)
        optimizer.zero_grad()
        loss = loss_function(logits_batch, label_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_losses.append(loss.item())

    epoch_losses.append(epoch_loss / len(trainloader))
    print(f', loss {epoch_losses[-1]}')

    # if epoch % 25 == 0:
    #     #  Book-keeping and visualizing every tenth iterations
    #     with torch.no_grad():
    #         lgt = model(im.unsqueeze(0).to(device, dtype=torch.float))
    #         test_loss = 0
    #         for batch in valloader:
    #             image_batch, label_batch = batch  # unpack the data
    #             image_batch = image_batch.to(device, dtype=torch.float)
    #             label_batch = label_batch.to(device, dtype=torch.long)
    #             logits_batch = model(image_batch)
    #             loss = loss_function(logits_batch, label_batch)
    #             test_loss += loss.item()
    #
    #             prob_pred = torch.nn.Softmax(dim=1)(logits_batch)
    #             pred = prob_pred.argmax(axis=1)
    #             intersection = torch.logical_and(label_batch, pred)
    #             union = torch.logical_or(label_batch, pred)
    #             iou_score = torch.sum(intersection) / torch.sum(union)
    #
    #         test_losses.append(test_loss / len(valloader))
    #         iou_scores.append(iou_score.item())
    #
    #     prob = torch.nn.Softmax(dim=1)(lgt)
    #
    #
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     ax[0].imshow(prob[0, 1].cpu().detach())
    #     ax[0].set_title(f'Prediction, epoch:{len(epoch_losses) - 1}')
    #
    #     ax[1].plot(np.linspace(0, len(epoch_losses), len(batch_losses)),
    #                batch_losses, lw=0.5)
    #     ax[1].plot(np.arange(len(epoch_losses)) + 0.5, epoch_losses, lw=2)
    #     ax[1].plot(np.linspace(9.5, len(epoch_losses) - 0.5, len(test_losses)),
    #                test_losses, lw=1)
    #     ax[1].plot(np.linspace(9.5, len(epoch_losses) - 0.5, len(iou_scores)),
    #                iou_scores, lw=1)
    #     ax[1].set_title('Batch loss, epoch loss (training) and test loss')
    #     ax[1].set_ylim(0, 1.1 * max(epoch_losses + test_losses))
    #     plt.show()

# %%  Show predictions for a few images from the test set.

idxs = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(3, len(idxs), figsize=(15, 10))

for n, idx in enumerate(idxs):
    im_val, lb_val = ValData[idx]
    with torch.no_grad():
        lgt_val = model(im_val.unsqueeze(0).to(device, dtype=torch.float))

    prob_val = torch.nn.Softmax(dim=1)(lgt_val)

    ax[0, n].imshow(im_val.permute(1, 2, 0))
    ax[1, n].imshow(lb_val)
    res = prob_val[0, 1].cpu().detach()
    # res[res > 0.5] = 1
    # res[res <= 0.5] = 0
    ax[2, n].imshow(res)
fig.suptitle('Test images, labels and predictions')
plt.show()
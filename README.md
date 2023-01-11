Code for DeformIrisNet: An Identity-Preserving Model of Iris Texture Deformation

To run the network to dilate a single pupil image, use single_image.py as follows:

> python single_image.py <weight_path> <small_pupil_image_path> <big_pupil_mask_image_path>

The weight is the .pth file provided. A large pupil image will be saved as dilated.png, play around with the mask and see the output.

The paper can be found here: 
> https://openaccess.thecvf.com/content/WACV2023/papers/Khan_DeformIrisNet_An_Identity-Preserving_Model_of_Iris_Texture_Deformation_WACV_2023_paper.pdf

> https://arxiv.org/abs/2207.08980

Bibtex:
```
@inproceedings{khan2023deformirisnet,
  title={DeformIrisNet: An Identity-Preserving Model of Iris Texture Deformation},
  author={Khan, Siamul Karim and Tinsley, Patrick and Czajka, Adam},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={900--908},
  year={2023}
}
```


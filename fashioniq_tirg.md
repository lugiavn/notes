# Run TIRG on Fashion-IQ dataset

The benchmark is described at https://sites.google.com/view/lingir/fashion-iq

Download the label at https://github.com/XiaoxiaoGuo/fashion-iq

Download the images at http://data.nam.ai/fashioniq_images.zip

Clone tirg code at https://github.com/google/tirg

Modify datasets.py to add the new dataset:

```python

class FashionIQ(BaseDataset):
  def __init__(self, path, split='train', transform=None):
    super(FashionIQ, self).__init__()
    
    self.split = split
    self.transform = transform
    self.img_path = path + '/'
    
    failures = [u'B00AZJJGDS', u'B00BTDROGU', u'B00AZJJER6', u'B00AZJJYBM', u'B009H3SDPK', u'B00AKQBDKU', u'B00AZJJE1C', u'B00BTDS338', u'B0082P4LLO', u'B00CM2RZ3E', u'B00C621JPK', u'B00AZGHX7M', u'B00A885VUI', u'B0057LRI6Q', u'B008583YKC', u'B009NY5YZA']

    data = {
        'image_splits': {},
        'captions': {}
    }
    import os
    for k in data:
        for f in os.listdir(path + '/' + k):
            if (split == 'train' and 'train' in f) or (split == 'test' and 'val' in f):
                d = json.load(open(path + '/' + k + '/' + f))
                data[k][f] = d

    imgs = []
    asin2id = {}
    for k in data['image_splits']:
        for asin in data['image_splits'][k]:
            if asin in failures:
                continue
            asin2id[asin] = len(imgs)
            imgs += [{
                'asin': asin,
                'file_path': path + '/images/' + asin + '.jpg',
                'captions': [asin2id[asin]]
            }]

    queries = []
    for k in data['captions']:
        for query in data['captions'][k]:
            if query['candidate'] in failures or query['target'] in failures:
                continue
            query['source_id'] = asin2id[query['candidate']]
            query['target_id'] = asin2id[query['target']]
            query['captions'] = [c.encode('utf-8') for c in query['captions']]
            queries += [query]
            
    
    self.data = data
    self.imgs = imgs
    self.queries = queries
    
    if split == 'test':
        self.test_queries = [{
              'source_img_id': query['source_id'],
              'target_img_id': query['target_id'],
              'target_caption': query['target_id'],
              'target_caption': query['target_id'],
              'mod': {'str': query['captions'][0] + ' inadditiontothat ' + query['captions'][1]}
          } for query in queries]

  def get_all_texts(self):
    texts = ['inadditiontothat']
    for query in self.queries:
        texts += query['captions']
    return texts

  def __len__(self):
    return len(self.imgs)

  def generate_random_query_target(self):
    query = random.choice(self.queries)
    mod_str = random.choice([
            query['captions'][0] + ' inadditiontothat ' + query['captions'][1],
            query['captions'][1] + ' inadditiontothat ' + query['captions'][0]
        ])
        
    return {
      'source_img_id': query['source_id'],
      'source_img_data': self.get_img(query['source_id']),
      'target_img_id': query['target_id'],
      'target_caption': query['target_id'],
      'target_img_data': self.get_img(query['target_id']),
      'target_caption': query['target_id'],
      'mod': {'str': mod_str}
    }

  def get_img(self, idx, raw_img=False):
    img_path = self.imgs[idx]['file_path']
    with open(img_path, 'rb') as f:
      img = PIL.Image.open(f)
      img = img.convert('RGB')
    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img
```

Also change main.py

```python
  elif opt.dataset == 'fashioniq':
    trainset = datasets.FashionIQ(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.FashionIQ(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
```

Now run:
```
python2 main.py --dataset=fashioniq --dataset_path=./fashioniq \
  --num_iters=160000 --model=concat --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=fashioniq_concat

python2 main.py --dataset=fashioniq --dataset_path=./fashioniq \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=fashioniq_tirg
```
Train+test progress is saved in ./runs/, use tensorboard to view the logs

On an average GPU, takes couple of hours to a day to see some good performance.
```
@inproceedings{vo2019composing,
  title={Composing Text and Image for Image Retrieval-An Empirical Odyssey},
  author={Vo, Nam and Jiang, Lu and Sun, Chen and Murphy, Kevin and Li, Li-Jia and Fei-Fei, Li and Hays, James},
  booktitle={CVPR},
  year={2019}
}
```

import os
import shutil
import json
from datasets.common import *

def get_default_transform(size=64):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat(3*[x]) if x.size(0)==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_augmentation_transform(size=64):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat(3*[x]) if x.size(0)==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_default_target_transform():
    transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
    return transform

class OfficeHome(MultiDomainDataset):
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    input_shape = (3, 64, 64)
    num_classes = 65
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        if not 'data_transform' in kwargs:
            kwargs['data_transform'] = get_default_transform()
        if not 'target_transform' in kwargs:
            kwargs['target_transform'] = get_default_target_transform()
        assert all(domain in OfficeHome.domains for domain in domains_to_use)
        root = os.path.join('.', 'downloads', 'OfficeHomeDataset')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC',
                                         os.path.join('.', 'downloads', 'temp.zip'))
            os.rename(os.path.join('.', 'downloads', 'OfficeHomeDataset_10072016'), root)
        super().__init__(root, domains_to_use, **kwargs)

class VLCS(MultiDomainDataset):
    domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    input_shape = (3, 64, 64)
    num_classes = 5
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        if not 'data_transform' in kwargs:
            kwargs['data_transform'] = get_default_transform()
        if not 'target_transform' in kwargs:
            kwargs['target_transform'] = get_default_target_transform()
        assert all(domain in VLCS.domains for domain in domains_to_use)
        root = os.path.join('.', 'downloads', 'VLCS')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8',
                                         os.path.join('.', 'downloads', 'temp.tar.gz'))
        super().__init__(root, domains_to_use, **kwargs)

class PACS(MultiDomainDataset):
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    input_shape = (3, 64, 64)
    num_classes = 7
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        if not 'data_transform' in kwargs:
            kwargs['data_transform'] = get_default_transform()
        if not 'target_transform' in kwargs:
            kwargs['target_transform'] = get_default_target_transform()
        root = os.path.join('.', 'downloads', 'PACS')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd',
                                         os.path.join('.', 'downloads', 'temp.zip'))
            os.rename(os.path.join('.', 'downloads', 'kfold'), root)
        super().__init__(root, domains_to_use, **kwargs)

class Sviro(MultiDomainDataset):
    domains = ['aclass', 'escape', 'hilux', 'i3', 'lexus', 'tesla', 'tiguan', 'tucson', 'x5', 'zoe']
    input_shape = (3, 64, 64)
    num_classes = 7
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        if not 'data_transform' in kwargs:
            kwargs['data_transform'] = get_default_transform()
        if not 'target_transform' in kwargs:
            kwargs['target_transform'] = get_default_target_transform()
        root = os.path.join('.', 'downloads', 'Sviro')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://sviro.kl.dfki.de/?wpdmdl=1731',
                                 os.path.join('.', 'downloads', 'temp.zip'))
            os.rename(os.path.join('.', 'downloads', 'SVIRO_DOMAINBED'), root)
        super().__init__(root, domains_to_use, **kwargs)

class DomainNet(MultiDomainDataset):
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    input_shape = (3, 64, 64)
    num_classes = 345
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        if not 'data_transform' in kwargs:
            kwargs['data_transform'] = get_default_transform()
        if not 'target_transform' in kwargs:
            kwargs['target_transform'] = get_default_target_transform()
        root = os.path.join('.', 'downloads', 'DomainNet')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            for idx, url in enumerate([
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
            ]):
                download_and_extract_dataset(url, os.path.join('.', 'downloads', 'DomainNet', 'temp_{}.zip'.format(idx)))
            download_file(r'https://github.com/facebookresearch/DomainBed/raw/main/domainbed/misc/domain_net_duplicates.txt',
                          os.path.join('.', 'downloads', 'DomainNet', 'duplicates.txt'))
            with open(os.path.join('.', 'downloads', 'DomainNet', 'duplicates.txt'), 'r') as F:
                for line in F.readlines():
                    try:
                        os.remove(os.path.join(root, line.strip()))
                    except OSError:
                        pass
        super().__init__(root, domains_to_use, **kwargs)

class TerraIncognita(MultiDomainDataset):
    domains = ['location_100', 'location_38', 'location_43', 'location_46']
    input_shape = (3, 64, 64)
    num_classes = 10
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        if not 'data_transform' in kwargs:
            kwargs['data_transform'] = get_default_transform()
        if not 'target_transform' in kwargs:
            kwargs['target_transform'] = get_default_target_transform()
        root = os.path.join('.', 'downloads', 'TerraIncognita')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            for idx, url in enumerate([
                r'https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz',
                r'https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip'
            ]):
                download_and_extract_dataset(
                    url, os.path.join('.', 'downloads', 'TerraIncognita', 'temp_{}'.format(idx)+'.tar.gz' if idx==0 else '.zip')
                )
        if all(f in os.listdir(root) for f in ['eccv_18_all_images_sm', 'caltech_images_20210113.json']):
            include_locations = ['38', '46', '100', '43']
            include_categories = [
                'bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', 'squirrel'
            ]
            images_folder = os.path.join(root, 'eccv_18_all_images_sm')
            annotations_file = os.path.join(root, 'caltech_images_20210113.json')
            destination_folder = root
            stats = {}
            with open(annotations_file, 'r') as F:
                data = json.load(F)
            category_dict = {}
            for item in data['categories']:
                category_dict[item['id']] = item['name']
            for image in data['images']:
                image_location = image['location']
                if image_location not in include_locations:
                    continue
                loc_folder = os.path.join(destination_folder, 'location_{}'.format(image_location))
                os.makedirs(loc_folder, exist_ok=True)
                image_id = image['id']
                image_fname = image['file_name']
                for annotation in data['annotations']:
                    if annotation['image_id'] == image_id:
                        if image_location not in stats.keys():
                            stats[image_location] = {}
                        category = category_dict[annotation['category_id']]
                        if category not in include_categories:
                            continue
                        if category not in stats[image_location]:
                            stats[image_location][category] = 0
                        else:
                            stats[image_location][category] += 1
                        loc_cat_folder = os.path.join(loc_folder, category)
                        os.makedirs(loc_cat_folder, exist_ok=True)
                        dst_path = os.path.join(loc_cat_folder, image_fname)
                        src_path = os.path.join(images_folder, image_fname)
                        if os.path.exists(src_path):
                            shutil.move(src_path, dst_path)
            shutil.rmtree(images_folder)
            os.remove(annotations_file)
        super().__init__(root, domains_to_use, **kwargs)
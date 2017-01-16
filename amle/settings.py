import os

pj = lambda *paths: os.path.abspath(os.path.join(*paths))

package_root = os.path.abspath(os.path.dirname(__file__))
repo_root = pj(package_root, '../')

# data_root = 'Z:/automl_data_100k'
data_root = pj(repo_root, 'tmp', 'automl_data_100k')

default_working_dir = pj(repo_root, 'tmp')

if __name__ == '__main__':
    print package_root
    print repo_root

    import glob
    print glob.glob(data_root + '/*')

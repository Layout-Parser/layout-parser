from iopath.common.file_io import HTTPURLHandler
from iopath.common.file_io import PathManager as PathManagerBase

# A trick learned from https://github.com/facebookresearch/detectron2/blob/65faeb4779e4c142484deeece18dc958c5c9ad18/detectron2/utils/file_io.py#L3


class DropboxHandler(HTTPURLHandler):
    """
    Supports download and file check for dropbox links
    """

    def _get_supported_prefixes(self):
        return ["https://www.dropbox.com"]

    def _isfile(self, path):
        return path in self.cache_map


PathManager = PathManagerBase()
PathManager.register_handler(DropboxHandler())
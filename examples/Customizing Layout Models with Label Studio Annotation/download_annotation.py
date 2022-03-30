import pdf2image
import tempfile
import urllib.request
import pandas as pd
import zipfile

opener = urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

def download_auxiliary_paper_images(target_path: str = "downloaded-annotations"):

    data_to_download = pd.DataFrame(
        [
            ["1810.04805v2", 10, "1810.04805v2-10_ea8f.jpg"],
            ["1810.04805v2", 11, "1810.04805v2-11_213f.jpg"],
            ["1810.04805v2", 9, "1810.04805v2-9_dc05.jpg"],
            ["1908.03557v1", 10, "1908.03557v1-10_fa12.jpg"],
            ["1908.03557v1", 11, "1908.03557v1-11_a737.jpg"],
        ],
        columns=["arxiv_id", "page", "filename"],
    )

    for arxiv_id, gp in data_to_download.groupby("arxiv_id"):
        with tempfile.TemporaryDirectory() as tempdir:
            arxiv_link = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
            urllib.request.urlretrieve(arxiv_link, f"{tempdir}/{arxiv_id}.pdf")
            pdf_images = pdf2image.convert_from_path(
                f"{tempdir}/{arxiv_id}.pdf", dpi=72
            )
            for _, row in gp.iterrows():
                pdf_images[row["page"]].save(f"{target_path}/images/{row['filename']}")


ANNOTATION_FILE_PATH = "http://szj.io/assets/files/data/layoutparser-webinar-annotations-2022-Feb.zip"

def download_zipped_annotations(): 
    filehandle, _ = urllib.request.urlretrieve(ANNOTATION_FILE_PATH)
    zip_ref = zipfile.ZipFile(filehandle, 'r')
    zip_ref.extractall("./") # extract file to dir
    zip_ref.close() # close file

if __name__ == "__main__":
    download_zipped_annotations()
    download_auxiliary_paper_images()
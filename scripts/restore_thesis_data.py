"""Recover only the historical Gray dataset from the preserved Git tag."""
import io
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parents[1]
archive=subprocess.check_output(['git','archive','v2022-thesis','Gray'],cwd=root)
destination=root/'data'/'original'
destination.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    for member in tar.getmembers():
        path=destination/member.name
        if not path.resolve().is_relative_to(destination.resolve()) or member.issym() or member.islnk():
            raise ValueError('Unexpected archive member')
    tar.extractall(destination,filter='data')
print('Restored historical Gray images; consult data/original/README.md for provenance limits.')

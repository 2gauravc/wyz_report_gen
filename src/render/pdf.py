import subprocess
import shutil
import glob
import os 

def convert_htmls_to_pdfs(html_out_dir: str, pdf_out_dir: str, wkhtmltopdf_path: str | None = None) -> None:
    """Convert all .html files in html_out_dir to .pdf using wkhtmltopdf and write PDFs to pdf_out_dir."""
    if wkhtmltopdf_path is None:
        wkhtmltopdf_path = shutil.which('wkhtmltopdf')
    if not wkhtmltopdf_path:
        raise FileNotFoundError('wkhtmltopdf not found on PATH; please install it (apt-get install wkhtmltopdf)')

    html_files = sorted(glob.glob(os.path.join(html_out_dir, '*.html')))
    if not html_files:
        print(f'No HTML files found in {html_out_dir} to convert.')
        return

    os.makedirs(pdf_out_dir, exist_ok=True)

    for html in html_files:
        pdf_fname = os.path.splitext(os.path.basename(html))[0] + '.pdf'
        pdf_path = os.path.join(pdf_out_dir, pdf_fname)
        try:
            subprocess.run([wkhtmltopdf_path, "--enable-local-file-access", html, pdf_path],check=True)
            print(f'Converted: {html} -> {pdf_path}')
        except subprocess.CalledProcessError as e:
            print(f'Error converting {html}: {e}')

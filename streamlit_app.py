"""Aplikasi web prediksi langganan deposito berjangka.

Aplikasi ini TIDAK melatih model. Aplikasi hanya memuat Pipeline hasil
notebook (model_terpilih.joblib) dan memanggil predict().

Menjalankan di komputer sendiri:  streamlit run app.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Prediksi Deposito Berjangka', layout='centered')


@st.cache_resource
def muat_artefak():
    """Dimuat satu kali saja selama aplikasi hidup, bukan setiap interaksi."""
    model = joblib.load('model_terpilih.joblib')
    with open('metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    return model, meta


model, meta = muat_artefak()

st.title('Prediksi Langganan Deposito Berjangka')
st.caption(
    f"Model: {meta['nama_model']} · dilatih {meta['tanggal_dibuat']} · "
    f"scikit-learn {meta['versi_sklearn']}"
)

with st.expander('Tentang model ini'):
    st.write(
        'Model memperkirakan apakah seorang nasabah akan berlangganan deposito '
        'berjangka setelah dihubungi lewat telepon. Dilatih pada dataset Bank '
        'Marketing (Moro, Rita & Cortez, 2014, UCI Machine Learning Repository, '
        'lisensi CC BY 4.0).'
    )
    if 'metrik_uji' in meta:
        st.write('Kinerja pada data uji:', meta['metrik_uji'])
    st.warning(
        'Keluaran model adalah dugaan statistik, bukan kepastian, dan tidak '
        'boleh dipakai sebagai dasar tunggal keputusan terhadap nasabah.'
    )

# ---------------------------------------------------------------- formulir
st.sidebar.header('Masukkan data nasabah')
isian = {}

for kolom in meta['fitur_numerik']:
    bawah, atas = meta['rentang_numerik'][kolom]
    bulat = float(bawah).is_integer() and float(atas).is_integer()
    if bulat:
        isian[kolom] = st.sidebar.number_input(
            kolom, min_value=int(bawah), max_value=int(atas),
            value=int((bawah + atas) // 2), step=1,
        )
    else:
        isian[kolom] = st.sidebar.number_input(
            kolom, min_value=float(bawah), max_value=float(atas),
            value=float((bawah + atas) / 2), step=0.01, format='%.3f',
        )

for kolom in meta['fitur_kategorik']:
    isian[kolom] = st.sidebar.selectbox(kolom, meta['nilai_kategorik'][kolom])

# Baris paling penting di berkas ini: urutan kolom disusun ulang agar sama
# persis dengan urutan saat model dilatih.
baris = pd.DataFrame([isian])[meta['kolom_input']]

st.subheader('Data yang akan diprediksi')
st.dataframe(baris, use_container_width=True)

if st.button('Prediksi', type='primary'):
    hasil = model.predict(baris)[0]
    if str(hasil) == 'yes':
        st.success('Hasil prediksi: **yes** — nasabah diperkirakan berlangganan.')
    else:
        st.info('Hasil prediksi: **no** — nasabah diperkirakan tidak berlangganan.')

    if hasattr(model, 'predict_proba'):
        peluang = model.predict_proba(baris)[0]
        tabel = pd.DataFrame(
            {'Kelas': model.classes_.astype(str), 'Peluang': peluang.round(4)}
        ).set_index('Kelas')
        st.subheader('Peluang tiap kelas')
        st.bar_chart(tabel)
        st.caption(
            'Peluang Naive Bayes sering sangat dekat ke 0 atau 1. Angka itu '
            'menunjukkan keyakinan model, bukan probabilitas sebenarnya.'
        )

# ------------------------------------------------- prediksi banyak baris
st.divider()
st.subheader('Uji banyak baris sekaligus')
st.write(
    'Unggah `lima_baris_uji.csv` dari notebook untuk memeriksa bahwa prediksi '
    'aplikasi sama dengan prediksi Colab (Tahap 10).'
)

berkas = st.file_uploader('Berkas CSV', type='csv')
if berkas is not None:
    data = pd.read_csv(berkas)
    kurang = [k for k in meta['kolom_input'] if k not in data.columns]
    if kurang:
        st.error(f'Kolom berikut tidak ada di berkas: {kurang}')
    else:
        siap = data[meta['kolom_input']]
        data['y_prediksi_aplikasi'] = model.predict(siap)
        st.dataframe(data, use_container_width=True)

        if 'y_prediksi_colab' in data.columns:
            cocok = (data['y_prediksi_colab'] == data['y_prediksi_aplikasi']).all()
            if cocok:
                st.success('Seluruh prediksi aplikasi sama dengan prediksi Colab.')
            else:
                st.error(
                    'Ada prediksi yang berbeda. Periksa urutan kolom, tipe data, '
                    'dan apakah berkas model yang diunggah sudah versi terbaru.'
                )

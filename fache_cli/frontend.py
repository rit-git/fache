import hydra
from omegaconf import DictConfig

import streamlit as st
import requests

CLAIM = '明るく清潔感最高！東京駅の近くにあって良い。'
EVIDENCE = 'お部屋は壁紙などもオシャレで明るく、清潔感もあり、潔癖症の自分でも快適に過ごせました！ハウステンボスから疲れて帰って、足は痛くなってましたが、大浴場で体も軽くなりぐっすり眠ることができました＾＿＾朝食も品数が多くついつい食べ過ぎてしまいました。美味しかったです。冬のハウステンボスも素敵ですが、次回は春のチューリップまつりのハウステンボスに行きたいですね。ありがとうございました＾＿＾'

def fact_check(port):
    response = requests.post(
        f'http://localhost:{port}/api/predict',
        json={
            'claim': st.session_state['claim'],
            'para': st.session_state['evidence']
        }
    )
    if response.status_code == 200:
        response = response.json()
        result = []
        for item in response:
            result.append('【{}。確信度：{:.4f}】{} '.format('事実' if item['label'] else 'ウソ', item['score'], item['sent']))
        st.session_state['result'] = '\n'.join(result)

def update_textarea(key):
    st.session_state[key] = st.session_state[key]

@hydra.main(config_path="../conf", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    port=cfg.server.port if 'server' in cfg else 12345

    st.set_page_config(
        page_title="Fache",
        layout="wide",
    )
    st.title('事実性判定器　Fache')

    if 'evidence' not in st.session_state:
        st.session_state['evidence'] = EVIDENCE
    if 'claim' not in st.session_state:
        st.session_state['claim'] = CLAIM
    if 'result' not in st.session_state:
        st.session_state['result'] = ''

    st.text_area(
        '証拠となる文章をこちらに入力してください。',
        key='evidence',
        height=200,
        on_change=update_textarea,
        kwargs={'key': 'evidence'}
    )
    st.text_area(
        '判定対象とする文章をこちらに入力してください。',
        key='claim',
        height=200,
        on_change=update_textarea,
        kwargs={'key': 'claim'}
    )
    st.text_area(
        '判定結果',
        key='result',
        height=200
    )
    st.button(
        '事実性を判定！',
        on_click=fact_check,
        kwargs={
            'port': port
        }
    )


if __name__ == '__main__':
    main()
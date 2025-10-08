import streamlit as st
import pandas as pd
import numpy as np
import json
from google import genai
from google.genai.errors import APIError
import io

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Hiệu quả Dự án Kinh doanh",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Hiệu quả Dự án Đầu tư 📊")

# Lấy API Key từ Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- 1. HÀM TRÍCH XUẤT DỮ LIỆU BẰNG GEMINI (Sử dụng Structured Output) ---
def extract_financial_data(doc_content, api_key):
    """Sử dụng Gemini để trích xuất các thông số tài chính vào cấu trúc JSON."""
    if not api_key:
        return None, "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình Streamlit Secrets."

    try:
        client = genai.Client(api_key=api_key)
        
        # 1. Định nghĩa JSON Schema (Cấu trúc dữ liệu đầu ra)
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "initial_investment": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu (30,000,000,000 VND)"},
                "project_life": {"type": "INTEGER", "description": "Vòng đời dự án (ví dụ: 10 năm)"},
                "annual_revenue": {"type": "NUMBER", "description": "Doanh thu hàng năm (ví dụ: 3,500,000,000 VND)"},
                "annual_cost": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (ví dụ: 2,000,000,000 VND)"},
                "wacc": {"type": "NUMBER", "description": "WACC/Lãi suất chiết khấu (dạng thập phân, ví dụ: 0.13 cho 13%)"},
                "tax_rate": {"type": "NUMBER", "description": "Thuế suất TNDN (dạng thập phân, ví dụ: 0.2 cho 20%)"}
            },
            "required": ["initial_investment", "project_life", "annual_revenue", "annual_cost", "wacc", "tax_rate"]
        }

        # 2. Xây dựng Prompt và System Instruction
        system_instruction = (
            "Bạn là trợ lý trích xuất dữ liệu tài chính. Nhiệm vụ của bạn là đọc nội dung văn bản "
            "phương án kinh doanh và trích xuất 6 thông số chính xác theo cấu trúc JSON được cung cấp. "
            "Đảm bảo WACC và Thuế suất được chuyển đổi sang dạng thập phân (ví dụ: 13% -> 0.13). "
            "Bỏ qua các thông tin khác như Tài sản đảm bảo."
        )

        prompt = f"Trích xuất các thông số tài chính sau từ văn bản dưới đây:\n\n---\n{doc_content}\n---"

        # 3. Gọi API với Structured Output
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            system_instruction=system_instruction,
            config={"response_mime_type": "application/json", "response_schema": response_schema}
        )
        
        # 4. Phân tích kết quả JSON
        json_data = json.loads(response.text)
        
        # Chuẩn hóa WACC và Thuế suất (nếu người dùng nhập số lớn hơn 1)
        if json_data['wacc'] > 1:
            json_data['wacc'] /= 100
        if json_data['tax_rate'] > 1:
            json_data['tax_rate'] /= 100

        return json_data, None

    except APIError as e:
        return None, f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except json.JSONDecodeError:
        return None, "Lỗi phân tích JSON: AI không trả về đúng định dạng JSON. Vui lòng kiểm tra lại nội dung file."
    except Exception as e:
        return None, f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}"

# --- 2 & 3. HÀM TÍNH TOÁN DÒNG TIỀN VÀ CHỈ SỐ ---
@st.cache_data
def calculate_metrics(params):
    """Xây dựng bảng dòng tiền và tính toán NPV, IRR, PP, DPP."""
    I0 = params['initial_investment']
    N = params['project_life']
    R = params['annual_revenue']
    C = params['annual_cost']
    t = params['tax_rate']
    r = params['wacc']
    
    # 1. Tính Dòng tiền Hàng năm (Giả định Khấu hao = 0)
    EBIT = R - C
    EAT = EBIT * (1 - t) # Lợi nhuận sau thuế
    FCF = EAT # Dòng tiền thuần hàng năm (Free Cash Flow)

    # 2. Xây dựng Bảng Dòng tiền
    years = list(range(0, N + 1))
    cash_flows = [0] * (N + 1)
    cash_flows[0] = -I0 # Vốn đầu tư ban đầu
    for i in range(1, N + 1):
        cash_flows[i] = FCF

    df_cf = pd.DataFrame({
        'Năm': years,
        'Doanh thu (R)': [0] + [R] * N,
        'Chi phí (C)': [0] + [C] * N,
        'Lợi nhuận trước thuế (EBIT)': [0] + [EBIT] * N,
        'Thuế (20%)': [0] + [EBIT * t] * N,
        'Lợi nhuận sau thuế (EAT/FCF)': [0] + [FCF] * N,
        'Dòng tiền thuần (CF)': cash_flows,
    })
    
    # Định dạng các cột tiền tệ
    currency_cols = ['Doanh thu (R)', 'Chi phí (C)', 'Lợi nhuận trước thuế (EBIT)', 'Thuế (20%)', 'Lợi nhuận sau thuế (EAT/FCF)', 'Dòng tiền thuần (CF)']
    for col in currency_cols:
         df_cf[col] = df_cf[col].apply(lambda x: f"{x:,.0f} VND" if x != 0 else 0)

    # 3. Tính toán Chỉ số Tài chính
    
    # A. NPV
    npv_value = np.npv(r, cash_flows)
    
    # B. IRR (numpy.irr yêu cầu kết quả trả về là số)
    # Cần dùng cash_flows gốc (không phải cash_flows đã được định dạng chuỗi)
    try:
        irr_value = np.irr([cash_flows[0]] + [FCF] * N)
    except Exception:
        irr_value = np.nan # Không thể tính nếu dòng tiền không đổi dấu

    # C. PP (Payback Period - Thời gian hoàn vốn)
    # Công thức: PP = Năm trước khi hoàn vốn + (Vốn còn lại chưa thu hồi / FCF năm hoàn vốn)
    pp_value = I0 / FCF if FCF > 0 else float('inf')
    
    # D. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    pv_cash_flows = [cash_flows[0]] 
    for t_idx in range(1, N + 1):
        pv_cash_flows.append(FCF / (1 + r)**t_idx)

    cumulative_pv = np.cumsum(pv_cash_flows)
    
    dpp_value = float('inf')
    
    # Tìm năm hoàn vốn (năm mà cumulative_pv chuyển sang dương)
    try:
        if cumulative_pv.max() > 0:
            payback_year = np.where(cumulative_pv >= 0)[0][0]
            if payback_year == 0:
                 dpp_value = 0
            else:
                # Giá trị tích lũy PV cuối năm trước khi hoàn vốn (là số âm)
                prev_cumulative_pv = cumulative_pv[payback_year - 1]
                # Giá trị PV dòng tiền năm hoàn vốn
                pv_current_year = pv_cash_flows[payback_year]
                
                # DPP = Năm trước + (Giá trị tuyệt đối vốn còn lại / PV dòng tiền năm hoàn vốn)
                dpp_value = (payback_year - 1) + (abs(prev_cumulative_pv) / pv_current_year)
        elif cumulative_pv.max() < 0:
            dpp_value = float('inf') # Không bao giờ hoàn vốn
    except:
        dpp_value = float('inf')


    metrics = {
        'NPV': npv_value,
        'IRR': irr_value,
        'PP': pp_value,
        'DPP': dpp_value
    }

    return df_cf.iloc[1:], metrics

# --- 4. HÀM PHÂN TÍCH CHỈ SỐ BẰNG GEMINI ---
def get_ai_financial_analysis(metrics_data, api_key):
    """Yêu cầu AI phân tích các chỉ số hiệu quả dự án."""
    if not api_key:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."

    npv = metrics_data['NPV']
    irr = metrics_data['IRR']
    pp = metrics_data['PP']
    dpp = metrics_data['DPP']
    
    # Xử lý giá trị IRR vô hạn hoặc không xác định
    irr_display = f"{irr*100:.2f}%" if not np.isnan(irr) and irr != float('inf') and irr != float('-inf') else "Không xác định/Không thể tính"
    
    # Xử lý giá trị hoàn vốn vô hạn
    pp_display = f"{pp:.2f} năm" if pp != float('inf') else "Không hoàn vốn trong vòng đời dự án"
    dpp_display = f"{dpp:.2f} năm" if dpp != float('inf') else "Không hoàn vốn có chiết khấu trong vòng đời dự án"


    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính đầu tư có kinh nghiệm. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tính khả thi của dự án. 

    Chỉ số Đánh giá Hiệu quả Dự án:
    1. NPV (Giá trị Hiện tại Thuần): {npv:,.0f} VND
    2. IRR (Tỷ suất Sinh lời Nội tại): {irr_display}
    3. PP (Thời gian Hoàn vốn): {pp_display}
    4. DPP (Thời gian Hoàn vốn có Chiết khấu): {dpp_display}

    Phân tích tập trung vào việc dự án có nên được chấp nhận hay không, dựa trên quy tắc: NPV > 0 và so sánh IRR với WACC (đã được sử dụng để tính toán NPV).
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# ==============================================================================
# --- LOGIC GIAO DIỆN STREAMLIT ---
# ==============================================================================

st.info("💡 **LƯU Ý:** Để AI trích xuất dữ liệu chính xác nhất, vui lòng tải lên file **Markdown (.md)** hoặc **Text (.txt)** chứa các thông số: Vốn đầu tư, Vòng đời, Doanh thu, Chi phí, WACC, Thuế.")

# --- 1. Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải lên file Phương án Kinh doanh (.txt, .md, hoặc .docx)",
    type=['txt', 'md', 'docx']
)

if uploaded_file is not None:
    # Đọc nội dung file
    try:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        
        # Nếu là file DOCX, ta chỉ đọc nội dung thô (có thể không hoàn hảo)
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
             st.warning("Đã tải file DOCX. Quá trình trích xuất có thể không chính xác do cấu trúc phức tạp. Vui lòng kiểm tra kỹ kết quả.")
             # Thử đọc nội dung thô của file để AI xử lý
             doc_content = uploaded_file.getvalue().decode("utf-8", errors='ignore')
        else:
            # Đọc file Text/Markdown
            doc_content = uploaded_file.getvalue().decode("utf-8")
        
        st.subheader("Nội dung File đã tải lên (Tham chiếu)")
        with st.expander("Xem nội dung file"):
             st.text(doc_content[:3000] + "..." if len(doc_content) > 3000 else doc_content)

        # --- Lọc dữ liệu bằng AI ---
        st.markdown("---")
        if st.button("2. 🤖 Trích xuất Thông số Tài chính (AI)", type="primary"):
            if not API_KEY:
                st.error("Vui lòng cấu hình Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng AI.")
            else:
                with st.spinner("Đang gửi nội dung file tới Gemini để trích xuất dữ liệu..."):
                    params, error = extract_financial_data(doc_content, API_KEY)
                
                if error:
                    st.error(error)
                    st.session_state['extracted_params'] = None
                elif params:
                    st.session_state['extracted_params'] = params
                    st.success("Trích xuất dữ liệu thành công!")

    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}. Đảm bảo file không bị lỗi định dạng.")
        st.session_state['extracted_params'] = None

# --- 3. Hiển thị thông số đã trích xuất ---
if 'extracted_params' in st.session_state and st.session_state['extracted_params'] is not None:
    params = st.session_state['extracted_params']
    
    st.subheader("3. Thông số Dự án đã trích xuất")
    
    # Hiển thị tham số
    col_i0, col_n, col_r = st.columns(3)
    col_c, col_wacc, col_tax = st.columns(3)

    col_i0.metric("Vốn đầu tư ($I_0$)", f"{params['initial_investment']:,.0f} VND")
    col_n.metric("Vòng đời Dự án ($N$)", f"{params['project_life']} năm")
    col_r.metric("Doanh thu ($R$)", f"{params['annual_revenue']:,.0f} VND")
    col_c.metric("Chi phí ($C$)", f"{params['annual_cost']:,.0f} VND")
    col_wacc.metric("WACC ($r$)", f"{params['wacc']*100:.2f}%")
    col_tax.metric("Thuế suất ($t$)", f"{params['tax_rate']*100:.2f}%")
    
    # --- 4. Xây dựng Dòng tiền & Tính toán Chỉ số ---
    
    # Kích hoạt tính toán
    try:
        df_cf, metrics = calculate_metrics(params)
        st.session_state['cash_flow_table'] = df_cf
        st.session_state['financial_metrics'] = metrics
        
        st.markdown("---")
        st.subheader("4. Bảng Dòng tiền (Cash Flow) Hàng năm")
        st.dataframe(
            df_cf, 
            hide_index=True,
            use_container_width=True
        )
        
        # Hiển thị Chỉ số Hiệu quả
        st.markdown("---")
        st.subheader("5. Các Chỉ số Đánh giá Hiệu quả Dự án")
        
        col_npv, col_irr, col_pp, col_dpp = st.columns(4)

        # NPV
        npv_color = "green" if metrics['NPV'] > 0 else "red"
        col_npv.markdown(f"<p style='text-align: center; font-size: 16px; color: {npv_color};'>**NPV**</p>", unsafe_allow_html=True)
        col_npv.metric("", f"{metrics['NPV']:,.0f}", delta_color=npv_color, help="NPV > 0: Chấp nhận dự án. NPV < 0: Từ chối dự án.")
        col_npv.markdown(f"<p style='text-align: center; color: {npv_color}; font-size: 10px;'>VND</p>", unsafe_allow_html=True)
        
        # IRR
        irr_display = f"{metrics['IRR']*100:.2f}%" if not np.isnan(metrics['IRR']) and metrics['IRR'] != float('inf') else "N/A"
        irr_color = "green" if not np.isnan(metrics['IRR']) and metrics['IRR'] > params['wacc'] else "red"
        col_irr.markdown(f"<p style='text-align: center; font-size: 16px; color: {irr_color};'>**IRR**</p>", unsafe_allow_html=True)
        col_irr.metric("", irr_display, delta_color="off", help="IRR > WACC: Chấp nhận dự án.")
        
        # PP
        pp_display = f"{metrics['PP']:.2f}" if metrics['PP'] != float('inf') else "> N"
        pp_color = "green" if metrics['PP'] <= params['project_life'] else "red"
        col_pp.markdown(f"<p style='text-align: center; font-size: 16px; color: {pp_color};'>**PP (Hoàn vốn)**</p>", unsafe_allow_html=True)
        col_pp.metric("", f"{pp_display} năm", delta_color="off", help="PP < N: Dự án hoàn vốn trong vòng đời.")
        
        # DPP
        dpp_display = f"{metrics['DPP']:.2f}" if metrics['DPP'] != float('inf') else "> N"
        dpp_color = "green" if metrics['DPP'] <= params['project_life'] else "red"
        col_dpp.markdown(f"<p style='text-align: center; font-size: 16px; color: {dpp_color};'>**DPP (Hoàn vốn CK)**</p>", unsafe_allow_html=True)
        col_dpp.metric("", f"{dpp_display} năm", delta_color="off", help="DPP < N: Dự án hoàn vốn có chiết khấu trong vòng đời.")


        # --- 5. Yêu cầu AI Phân tích ---
        st.markdown("---")
        if st.button("6. Yêu cầu AI Phân tích Hiệu quả Dự án"):
             if not API_KEY:
                st.error("Vui lòng cấu hình Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng AI.")
             else:
                with st.spinner('Đang gửi các chỉ số tới Gemini để phân tích...'):
                    ai_result = get_ai_financial_analysis(metrics, API_KEY)
                
                st.markdown("#### 🤖 Kết quả Phân tích từ Gemini AI:")
                st.info(ai_result)

    except ZeroDivisionError:
        st.error("Lỗi Chia cho 0: Dòng tiền thuần (FCF) hàng năm bằng 0. Không thể tính PP và DPP.")
    except Exception as e:
        st.error(f"Lỗi tính toán: {e}. Vui lòng kiểm tra lại các thông số trích xuất.")

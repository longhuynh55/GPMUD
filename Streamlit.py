try:
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import os
    import numpy as np
except ImportError as e:
    print(f"Thiếu thư viện: {str(e)}")
    print("Vui lòng cài đặt các thư viện cần thiết bằng lệnh:")
    print("pip install streamlit pandas plotly numpy plotly-express")
    exit(1)

def load_csv_files(folder_path):
    """Load all CSV files from the specified folder"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return sorted(csv_files)

def load_stock_data(file_path):
    """Load and process stock data from CSV file"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

def create_candlestick_chart(df, symbol):
    """Create candlestick chart using Plotly"""
    # Convert datetime to numpy array to avoid warning
    dates = np.array(df['Date'])
    
    fig = go.Figure(data=[go.Candlestick(
        x=dates,
        open=df['Price Open'],
        high=df['Price High'],
        low=df['Price Low'],
        close=df['Price Close'],
        name=symbol
    )])

    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_white',
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            ),
            type="date"
        )
    )
    return fig

def main():
    st.title('Stock Price Viewer')
    
    # Đường dẫn tới thư mục chứa file CSV
    folder_path = "D:\dai hoc\GPM\data\processed_data\Data cleanned"  # Thay đổi đường dẫn này theo thư mục của bạn
    
    try:
        # Load danh sách file CSV
        all_csv_files = load_csv_files(folder_path)
        
        if not all_csv_files:
            st.error("Không tìm thấy file CSV nào trong thư mục!")
            return
        
        # Thanh tìm kiếm
        search_term = st.text_input("Tìm kiếm mã cổ phiếu:", "")
        
        # Lọc danh sách file dựa trên từ khóa tìm kiếm
        filtered_files = [f for f in all_csv_files if search_term.lower() in f.lower()]
        
        if not filtered_files:
            st.warning("Không tìm thấy cổ phiếu phù hợp!")
            return
            
        # Dropdown để chọn file từ danh sách đã lọc
        selected_file = st.selectbox(
            "Chọn mã cổ phiếu:", 
            filtered_files,
            key="stock_selector"
        )
        
        if selected_file:
            # Lấy tên cổ phiếu từ tên file
            symbol = os.path.splitext(selected_file)[0]
            
            # Load và xử lý dữ liệu
            df = load_stock_data(os.path.join(folder_path, selected_file))
            
            # Tạo và hiển thị biểu đồ
            fig = create_candlestick_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị thông tin tổng quan
            with st.expander("Thông tin chi tiết"):
                st.subheader('Tổng quan')
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Ngày bắt đầu: {df['Date'].min().strftime('%Y-%m-%d')}")
                    st.write(f"Tổng số phiên: {len(df)}")
                with col2:
                    st.write(f"Ngày kết thúc: {df['Date'].max().strftime('%Y-%m-%d')}")
                
                # Hiển thị giá mới nhất
                st.subheader('Giá mới nhất')
                latest = df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Giá mở cửa", f"{latest['Price Open']:,.2f}")
                col2.metric("Giá cao nhất", f"{latest['Price High']:,.2f}")
                col3.metric("Giá thấp nhất", f"{latest['Price Low']:,.2f}")
                col4.metric("Giá đóng cửa", f"{latest['Price Close']:,.2f}")
                
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()
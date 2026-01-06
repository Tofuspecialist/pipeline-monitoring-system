with col_right:
    st.subheader("📈 Pipeline Analytics")
    if total_leaks > 0:
        st.markdown(f'<div style="background: #fef2f2; border-left: 5px solid #ef4444; padding: 1rem;"><b>🚨 EMERGENCY</b><br>{total_leaks} leak(s) detected.</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="background: #f8fafc; border-left: 5px solid #10b981; padding: 1rem;"><b>Network Health</b><br>• {len(data[~data["leak"]])} normal<br>• Avg P: {data["pressure"].mean():.2f} bar</div>', unsafe_allow_html=True)
    
    st.dataframe(data[["id", "line", "pressure", "flow", "leak"]], use_container_width=True)
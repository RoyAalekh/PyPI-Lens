import streamlit as st
import streamlit.components.v1 as components
from database import PackageDB
import requests

def create_app():
    st.set_page_config(
        page_title="PyPI Lens",
        layout="wide"
    )

    @st.cache_resource
    def init_db():
        return PackageDB()

    def is_valid_url(url):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def validate_urls(results):
        for result in results:
            result['homepage_valid'] = is_valid_url(result['homepage']) if result['homepage'] else False
            result['repo_valid'] = is_valid_url(result['repository']) if result['repository'] else False
        return results

    def paginate_results(results, page_size):
        page = st.session_state.get('page', 1)
        total_pages = (len(results) + page_size - 1) // page_size

        start_index = (page - 1) * page_size
        end_index = start_index + page_size

        prev_disabled = page == 1
        next_disabled = page == total_pages

        prev_button, page_info, next_button = st.columns([1, 2, 1])

        with prev_button:
            if st.button("Previous", key="prev", disabled=prev_disabled):
                st.session_state['page'] = page - 1

        with page_info:
            st.markdown(f"<div style='text-align: center; font-size: 1rem;'>Page {page} of {total_pages}</div>", unsafe_allow_html=True)

        with next_button:
            if st.button("Next", key="next", disabled=next_disabled):
                st.session_state['page'] = page + 1

        return results[start_index:end_index], total_pages

    def main():
        # Header
        st.markdown(
            """
            <div style="position: fixed; top: 0; left: 0; width: 100%; background-color: white; z-index: 999; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 1rem; text-align: center;">
                <div style="margin-left: 250px;">
                    <h1 style="color: #0366d6; margin: 0;">PyPI Package Search</h1>
                    <p style="font-size: 1rem; color: #586069; margin: 0;">Search and explore Python packages with semantic matching</p>
                </div>
            </div>
            <div style="height: 120px;"></div>
            """,
            unsafe_allow_html=True,
        )

        db = init_db()

        # Show database stats in sidebar
        stats = db.get_stats()
        with st.sidebar:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h2 style="color: #0366d6;">Database Statistics</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Total Packages", f"{stats['total_packages']:,}")
            if stats['last_updated']:
                st.metric("Last Updated", stats['last_updated'].strftime("%Y-%m-%d %H:%M"))

        # Search interface
        query = st.text_input(
            "Search packages",
            placeholder="Enter package name, description, or keywords...",
        )

        if query:
            with st.spinner("Searching packages..."):
                results = db.search(query)

                if results:
                    results = validate_urls(results)
                    st.markdown(
                        f"<h3 style='color: #0366d6;'>Found {len(results)} matching packages</h3>",
                        unsafe_allow_html=True,
                    )

                    page_size = 5
                    paginated_results, total_pages = paginate_results(results, page_size)

                    for result in paginated_results:
                        # Format downloads number
                        downloads = f"{result['downloads']:,}" if result['downloads'] else "Not available"

                        # Format match score as percentage
                        match_score = f"{result['similarity'] * 100:.1f}%"

                        html_content = f"""
                        <div style="padding: 1.5rem; border-radius: 8px; border: 1px solid #e1e4e8; margin: 1rem 0; background-color: white; transition: box-shadow 0.3s ease;">
                            <div style="color: #0366d6; font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">
                                {result['name']} <small>v{result['version']}</small>
                            </div>

                            <div style="color: #586069; font-size: 1rem; font-style: italic; margin-bottom: 1rem;">
                                <span style="background-color: #28a745; color: white; padding: 0.3rem 0.7rem; border-radius: 4px; font-size: 0.95rem; font-weight: bold;">Match: {match_score}</span>
                                <span style="margin-left: 1rem;">{downloads} downloads</span>
                            </div>

                            <div style="margin: 0.5rem 0;">
                                {''.join([f'<span style="background-color: #f1f8ff; color: #0366d6; padding: 0.3rem 0.6rem; border-radius: 999px; font-size: 0.9rem; margin-right: 0.5rem;">{tag}</span>' for tag in result['tags']])}
                            </div>

                            <div style="margin-top: 1rem; display: flex; gap: 1rem; align-items: center;">
                                {f'<a href="{result["homepage"]}" target="_blank" style="text-decoration: none; color: #0366d6;">Homepage</a>' if result['homepage_valid'] else '<span style="color: #d73a49;">Homepage not available</span>'}
                                {f'<a href="{result["repository"]}" target="_blank" style="text-decoration: none; color: #0366d6;">Repository</a>' if result['repo_valid'] else '<span style="color: #d73a49;">Repository not available</span>'}
                            </div>
                        </div>
                        """

                        components.html(html_content, height=300)
                else:
                    st.info("No matching packages found. Try different search terms.")

        # Footer
        st.markdown(
            """
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: white; z-index: 999; box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1); padding: 1rem; text-align: center; font-size: 0.9rem; color: #586069; min-height: 80px;">
                <div style="margin-left: 250px;">
                    Developed by <a href="https://github.com/RoyAalekh" target="_blank" style="color: #0366d6; text-decoration: none;">AR</a>
                </div>
            </div>
            <div style="height: 80px;"></div>
            """,
            unsafe_allow_html=True,
        )

    return main


if __name__ == "__main__":
    app = create_app()
    app()

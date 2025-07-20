// React and ReactDOM are now loaded globally via CDN in index.html
const { useState, useEffect, useCallback, useRef } = React;

const API_BASE_URL = '/api';

function App() {
    // --- All states for the form inputs ---
    const [mode, setMode] = useState('full_auto');
    const [targetUrl, setTargetUrl] = useState('');
    const [maxProducts, setMaxProducts] = useState('');
    const [categoryPaths, setCategoryPaths] = useState('');
    const [proxies, setProxies] = useState('');
    const [headless, setHeadless] = useState(true);
    const [logLevel, setLogLevel] = useState('INFO');
    const [customSelectors, setCustomSelectors] = useState({
        name: '', price: '', image_url: '', description: '', brand: '', category: ''
    });
    const [paginationType, setPaginationType] = useState('auto');
    const [pageParamName, setPageParamName] = useState('page');
    const [nextButtonSelector, setNextButtonSelector] = useState('');
    const [autoAdjustConcurrency, setAutoAdjustConcurrency] = useState(true);
    const [autoAdjustMode, setAutoAdjustMode] = useState('full_auto');
    const [maxConcurrentBrowsersMin, setMaxConcurrentBrowsersMin] = useState(1);
    const [maxConcurrentBrowsersMax, setMaxConcurrentBrowsersMax] = useState(5);
    const [useRandomUserAgents, setUseRandomUserAgents] = useState(true);
    const [saveImages, setSaveImages] = useState(false);
    const [imageQuality, setImageQuality] = useState(85);
    const [imageFormat, setImageFormat] = useState('JPEG');
    const [maxEmptyCategoryAttempts, setMaxEmptyCategoryAttempts] = useState(10);
    const [domainFilter, setDomainFilter] = useState(true); // New: Filter external links

    // --- All other states ---
    const [scraperStatus, setScraperStatus] = useState({});
    const [recentLogs, setRecentLogs] = useState([]);
    const [scrapedProducts, setScrapedProducts] = useState([]);
    const [discoveredCategories, setDiscoveredCategories] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [modalContent, setModalContent] = useState('');
    const [modalTitle, setModalTitle] = useState('');
    const [strings, setStrings] = useState({});
    const logsEndRef = useRef(null); // Ref for auto-scrolling logs

    // --- Fetch strings.json for internationalization ---
    useEffect(() => {
        fetch('/strings.json')
            .then(response => response.json())
            .then(data => {
                setStrings(data);
                setIsLoading(false);
            })
            .catch(error => {
                console.error('Error loading strings:', error);
                setStrings({
                    loading: 'Loading...', ok: 'OK', close: 'Close',
                    dashboard_title: 'Advanced Web Scraper Dashboard',
                    scraper_config_title: 'Scraper Configuration',
                    start_error: 'Error starting scraper. Please check console.',
                    scraper_active_message: 'Scraper is currently active. Please wait or stop it first.',
                    target_url_required: 'Please enter a target URL.',
                    product_summary_title: 'Product Description Summary',
                    generating_summary: 'Generating summary... Please wait.',
                    no_summary_available: 'No summary available.',
                    no_description_for_summary: 'No description available for summarization.',
                    file_not_selected: 'No file selected.',
                    unsupported_file_format: 'Unsupported file format. Please upload JSON or CSV.',
                    error_processing_file: 'Error processing file:',
                    categories_imported_success: 'categories imported successfully.',
                    invalid_data_type: 'Invalid data type.',
                    no_data_for_csv: 'No data available for CSV export.',
                    csv_all_not_supported: 'CSV export for "all data" is not supported. Please select JSON.',
                    scraper_not_active: 'Scraper is not active.',
                    scraper_stopped_by_user: 'Scraper stopped by user.',
                    description_required: 'Description is required for summarization.',
                    error_summarizing: 'Error in summarization processing:',
                    scraper_status_idle: "Idle",
                    scraper_status_starting: "Starting",
                    scraper_status_running: "Running",
                    scraper_status_stopping: "Stopping",
                    scraper_status_stopped: "Stopped",
                    scraper_status_completed: "Completed",
                    scraper_status_fatal_error: "Fatal Error",
                    preparing_scraper: "Preparing scraper...",
                    a_fatal_error_occurred: "A fatal error occurred in the scraper."
                });
                setIsLoading(false);
            });
    }, []);

    // --- Auto-scroll logs to bottom ---
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [recentLogs]);

    // --- Fetch scraper status periodically ---
    const fetchScraperStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setScraperStatus(data);
            setRecentLogs(data.log_entries || []);
            setScrapedProducts(data.products || []);
            setDiscoveredCategories(data.categories || []);
        } catch (error) {
            console.error("Error fetching scraper status:", error);
            // Update status to indicate connection issue
            setScraperStatus(prev => ({
                ...prev,
                status: 'disconnected',
                message: `Connection error: ${error.message}`
            }));
        }
    }, []);

    useEffect(() => {
        fetchScraperStatus(); // Initial fetch
        const interval = setInterval(fetchScraperStatus, 3000); // Fetch every 3 seconds
        return () => clearInterval(interval); // Cleanup on component unmount
    }, [fetchScraperStatus]);

    // --- Format timestamp for display ---
    const formatTimestamp = (timestamp) => {
        if (!timestamp) return 'N/A';
        const date = new Date(timestamp * 1000); // Convert Unix timestamp to milliseconds
        return date.toLocaleString(); // Format to local date and time string
    };

    // --- Show modal for messages/summaries ---
    const showMessageModal = (title, content) => {
        setModalTitle(title);
        setModalContent(content);
        setShowModal(true);
    };

    // --- Handle Start Scraping ---
    const handleStartScraping = async () => {
        if (scraperStatus.status === 'running' || scraperStatus.status === 'starting') {
            showMessageModal(strings.scraper_status_title, strings.scraper_active_message);
            return;
        }
        if (!targetUrl) {
            showMessageModal(strings.scraper_config_title, strings.target_url_required);
            return;
        }

        const config = {
            mode,
            target_url: targetUrl,
            max_products_to_scrape: maxProducts ? parseInt(maxProducts) : null,
            categories_to_scrape: categoryPaths.split(',').map(s => s.trim()).filter(s => s),
            proxies: proxies.split(',').map(s => s.trim()).filter(s => s),
            headless,
            log_level: logLevel,
            product_detail_selectors: customSelectors,
            pagination_type: paginationType,
            page_param_name: pageParamName,
            next_button_selector: nextButtonSelector,
            auto_adjust_concurrency: autoAdjustConcurrency,
            auto_adjust_mode: autoAdjustMode,
            max_concurrent_browsers_min: parseInt(maxConcurrentBrowsersMin),
            max_concurrent_browsers_max: parseInt(maxConcurrentBrowsersMax),
            use_random_user_agents: useRandomUserAgents,
            save_images: saveImages,
            image_quality: parseInt(imageQuality),
            image_format: imageFormat,
            max_empty_category_attempts: parseInt(maxEmptyCategoryAttempts),
            domain_filter: domainFilter // Include new domain filter
        };

        try {
            const response = await fetch(`${API_BASE_URL}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || strings.start_error);
            }
            fetchScraperStatus(); // Update status immediately after starting
        } catch (error) {
            console.error("Error starting scraper:", error);
            showMessageModal(strings.scraper_status_title, `${strings.start_error} ${error.message}`);
        }
    };

    // --- Handle Stop Scraping ---
    const handleStopScraping = async () => {
        if (scraperStatus.status === 'idle' || scraperStatus.status === 'stopped' || scraperStatus.status === 'completed') {
            showMessageModal(strings.scraper_status_title, strings.scraper_not_active);
            return;
        }
        try {
            const response = await fetch(`${API_BASE_URL}/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || strings.scraper_stopped_by_user);
            }
            fetchScraperStatus(); // Update status immediately after stopping
        } catch (error) {
            console.error("Error stopping scraper:", error);
            showMessageModal(strings.scraper_status_title, `${strings.scraper_stopped_by_user} ${error.message}`);
        }
    };

    // --- Handle Summarize Description ---
    const handleSummarizeDescription = async (description) => {
        if (!description) {
            showMessageModal(strings.product_summary_title, strings.no_description_for_summary);
            return;
        }
        showMessageModal(strings.product_summary_title, strings.generating_summary);
        try {
            const response = await fetch(`${API_BASE_URL}/summarize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || strings.error_summarizing);
            }
            showMessageModal(strings.product_summary_title, data.summary || strings.no_summary_available);
        } catch (error) {
            console.error("Error summarizing description:", error);
            showMessageModal(strings.product_summary_title, `${strings.error_summarizing} ${error.message}`);
        }
    };

    // --- Handle Upload Categories ---
    const handleUploadCategories = async (event) => {
        const file = event.target.files[0];
        if (!file) {
            showMessageModal(strings.upload_categories_title, strings.file_not_selected);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/upload_categories`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `${strings.error_processing_file} ${file.name}`);
            }
            showMessageModal(strings.upload_categories_title, `${file.name}: ${data.message}`);
            fetchScraperStatus(); // Refresh status to show new categories
        } catch (error) {
            console.error("Error uploading categories:", error);
            showMessageModal(strings.upload_categories_title, `${strings.error_processing_file} ${error.message}`);
        }
    };

    // --- Handle Download Data ---
    const handleDownloadData = async (type, format) => {
        try {
            const response = await fetch(`${API_BASE_URL}/download_data?type=${type}&format=${format}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `Error downloading ${type} data.`);
            }
            const blob = await response.blob();
            const filename = response.headers.get('Content-Disposition').split('filename=')[1];
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showMessageModal(strings.download_data_title, `Data downloaded successfully as ${filename}.`);
        } catch (error) {
            console.error("Error downloading data:", error);
            showMessageModal(strings.download_data_title, `Error downloading data: ${error.message}`);
        }
    };

    // --- Helper Components for UI Structure ---

    // Modal Component
    const Modal = ({ title, content, onClose, strings }) => {
        if (!showModal) return null;
        return (
            <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
                <div className="bg-gray-800 rounded-lg shadow-xl max-w-lg w-full p-6 border border-gray-700">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-xl font-semibold text-gray-100">{title}</h3>
                        <button onClick={onClose} className="text-gray-400 hover:text-gray-100 transition-colors">
                            <i data-lucide="x" className="w-6 h-6"></i>
                        </button>
                    </div>
                    <div className="text-gray-300 mb-6 max-h-96 overflow-y-auto whitespace-pre-wrap">
                        {content}
                    </div>
                    <div className="flex justify-end">
                        <button
                            onClick={onClose}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md transition-colors shadow-md"
                        >
                            {strings.ok}
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    // Input Field Component
    const InputField = ({ label, type = 'text', value, onChange, placeholder, className = '', disabled = false, children }) => (
        <div className="mb-4">
            <label className="block text-gray-300 text-sm font-medium mb-2 flex items-center">
                {label}
                {children}
            </label>
            <input
                type={type}
                value={value}
                onChange={onChange}
                placeholder={placeholder}
                disabled={disabled}
                className={`shadow-sm appearance-none border border-gray-700 rounded-md w-full py-2 px-3 bg-gray-700 text-gray-100 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${className}`}
            />
        </div>
    );

    // Select Field Component
    const SelectField = ({ label, value, onChange, options, className = '', children }) => (
        <div className="mb-4">
            <label className="block text-gray-300 text-sm font-medium mb-2 flex items-center">
                {label}
                {children}
            </label>
            <div className="relative">
                <select
                    value={value}
                    onChange={onChange}
                    className={`block appearance-none w-full bg-gray-700 border border-gray-700 text-gray-100 py-2 px-3 pr-8 rounded-md shadow-sm leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${className}`}
                >
                    {options.map(option => (
                        <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                    <i data-lucide="chevron-down" className="w-4 h-4"></i>
                </div>
            </div>
        </div>
    );

    // Checkbox Toggle Component
    const CheckboxToggle = ({ label, checked, onChange, children }) => (
        <div className="mb-4 flex items-center">
            <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" checked={checked} onChange={onChange} className="sr-only peer" />
                <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:border-gray-300 after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                <span className="ml-3 text-sm font-medium text-gray-300 flex items-center">
                    {label}
                    {children}
                </span>
            </label>
        </div>
    );

    // Tooltip Component
    const Tooltip = ({ content }) => (
        <span className="relative group ml-2">
            <i data-lucide="help-circle" className="w-4 h-4 text-gray-400 cursor-help"></i>
            <span className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 hidden group-hover:block w-48 p-2 text-xs text-white bg-gray-700 rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                {content}
            </span>
        </span>
    );

    // Status Display Component
    const StatusDisplay = ({ status, strings, formatTimestamp }) => (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
            {Object.entries({
                [strings.status_label]: strings[`scraper_status_${status.status}`] || status.status,
                [strings.message_label]: status.message,
                [strings.products_scraped_label]: status.scraped_product_count,
                [strings.links_discovered_count_label]: status.total_product_links_found,
                [strings.categories_discovered_count_label]: status.discovered_categories_count,
                [strings.errors_label]: status.error_count,
                [strings.warnings_label]: status.warning_count,
                [strings.critical_errors_label]: status.critical_count,
                [strings.current_concurrency_label]: status.current_dynamic_concurrency,
                [strings.last_updated_label]: formatTimestamp(status.last_updated_at)
            }).map(([label, value]) => (
                <p key={label} className="text-sm">
                    <strong className="text-gray-200">{label}: </strong>
                    <span className={
                        label === strings.status_label ?
                            (status.status === 'running' ? 'text-green-400 font-semibold' :
                            status.status === 'stopping' || status.status === 'starting' ? 'text-yellow-400 font-semibold' :
                            status.status === 'fatal_error' ? 'text-red-400 font-semibold' :
                            'text-gray-300') :
                        label === strings.critical_errors_label && value > 0 ? 'text-red-400 font-semibold' :
                        label === strings.errors_label && value > 0 ? 'text-orange-400' :
                        label === strings.warnings_label && value > 0 ? 'text-yellow-400' :
                        'text-gray-300'
                    }>
                        {value || '0'}
                    </span>
                </p>
            ))}
        </div>
    );

    // Logs Display Component
    const LogsDisplay = ({ logs, strings, formatTimestamp, logsEndRef }) => (
        <div className="bg-gray-800 p-4 rounded-lg h-96 overflow-y-auto border border-gray-700 text-sm font-mono shadow-inner">
            {logs.length > 0 ? (
                logs.map((log, i) => (
                    <p key={i} className={`whitespace-pre-wrap mb-1 ${
                        log.level === 'ERROR' || log.level === 'CRITICAL' ? 'text-red-400' :
                        log.level === 'WARNING' ? 'text-orange-400' :
                        'text-gray-300'
                    }`}>
                        <span className="text-gray-500">[{formatTimestamp(log.timestamp)}]</span> {log.message}
                    </p>
                ))
            ) : (
                <p className="text-gray-500">{strings.no_logs_available}</p>
            )}
            <div ref={logsEndRef} /> {/* Scroll target */}
        </div>
    );

    // Data Preview Component (Generic)
    const DataPreview = ({ title, items, noItemsMessage, renderItem }) => (
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-lg flex-grow flex flex-col">
            <h2 className="text-xl font-semibold mb-4 text-gray-100">{title}</h2>
            <div className="overflow-y-auto flex-grow">
                {items.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {items.map((item, i) => (
                            <div key={i} className="bg-gray-900 p-4 rounded-md border border-gray-700 shadow-md">
                                {renderItem(item)}
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="text-gray-500 italic">{noItemsMessage}</p>
                )}
            </div>
        </div>
    );

    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-gray-900 text-gray-100">
                <p className="text-lg">{strings.loading}</p>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-6 space-y-8">
            <h1 className="text-4xl font-extrabold text-center text-blue-400 mb-8 tracking-wide">
                <i data-lucide="bot" className="w-10 h-10 inline-block mr-3"></i>
                {strings.dashboard_title}
            </h1>

            {/* Scraper Configuration Section */}
            <section className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700">
                <h2 className="text-2xl font-bold text-gray-100 mb-6 flex items-center">
                    <i data-lucide="settings" className="w-6 h-6 inline-block mr-3"></i>
                    {strings.scraper_config_title}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {/* Scraping Mode */}
                    <SelectField
                        label={strings.mode_label}
                        value={mode}
                        onChange={(e) => setMode(e.target.value)}
                        options={[
                            { value: 'full_auto', label: strings.mode_full_auto },
                            { value: 'manual_config', label: strings.mode_manual_config },
                            { value: 'single_product', label: strings.mode_single_product },
                        ]}
                    />

                    {/* Target URL */}
                    <InputField
                        label={strings.target_url_label}
                        type="url"
                        value={targetUrl}
                        onChange={(e) => setTargetUrl(e.target.value)}
                        placeholder={strings.target_url_placeholder}
                        className="col-span-1 md:col-span-2"
                    />

                    {/* Max Products to Scrape */}
                    <InputField
                        label={strings.max_products_label}
                        type="number"
                        value={maxProducts}
                        onChange={(e) => setMaxProducts(e.target.value)}
                        placeholder={strings.max_products_placeholder}
                    />

                    {/* Category Paths (Conditional) */}
                    {(mode === 'manual_config' || mode === 'full_auto') && (
                        <InputField
                            label={strings.category_paths_label}
                            value={categoryPaths}
                            onChange={(e) => setCategoryPaths(e.target.value)}
                            placeholder={strings.category_paths_placeholder}
                            className="col-span-1 md:col-span-2"
                        />
                    )}

                    {/* Proxies */}
                    <InputField
                        label={strings.proxies_label}
                        value={proxies}
                        onChange={(e) => setProxies(e.target.value)}
                        placeholder={strings.proxies_placeholder}
                        className="col-span-1 md:col-span-2"
                    />

                    {/* Headless Mode */}
                    <CheckboxToggle
                        label={strings.headless_mode_label}
                        checked={headless}
                        onChange={(e) => setHeadless(e.target.checked)}
                    >
                        <Tooltip content="Run the browser in the background without a visible UI. Recommended for performance." />
                    </CheckboxToggle>

                    {/* Log Level */}
                    <SelectField
                        label={strings.log_level_label}
                        value={logLevel}
                        onChange={(e) => setLogLevel(e.target.value)}
                        options={[
                            { value: 'DEBUG', label: strings.log_level_debug },
                            { value: 'INFO', label: strings.log_level_info },
                            { value: 'WARNING', label: strings.log_level_warning },
                            { value: 'ERROR', label: strings.log_level_error },
                            { value: 'CRITICAL', label: strings.log_level_critical },
                        ]}
                    />

                    {/* Domain Filter */}
                    <CheckboxToggle
                        label={strings.domain_filter_label}
                        checked={domainFilter}
                        onChange={(e) => setDomainFilter(e.target.checked)}
                    >
                        <Tooltip content="Only discover and scrape links within the target URL's domain. Prevents scraping external websites." />
                    </CheckboxToggle>
                </div>

                {/* Custom Selectors Section (Conditional) */}
                {mode === 'manual_config' && (
                    <div className="mt-8 pt-6 border-t border-gray-700">
                        <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
                            <i data-lucide="code" className="w-5 h-5 inline-block mr-2"></i>
                            {strings.custom_selectors_title}
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.keys(customSelectors).map(key => (
                                <InputField
                                    key={key}
                                    label={strings[`${key}_selector_label`]}
                                    value={customSelectors[key]}
                                    onChange={(e) => setCustomSelectors({ ...customSelectors, [key]: e.target.value })}
                                    placeholder={strings[`${key}_selector_placeholder`]}
                                />
                            ))}
                        </div>
                    </div>
                )}

                {/* Pagination Settings (Conditional) */}
                {mode !== 'single_product' && (
                    <div className="mt-8 pt-6 border-t border-gray-700">
                        <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
                            <i data-lucide="chevrons-right" className="w-5 h-5 inline-block mr-2"></i>
                            {strings.pagination_type_label}
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <SelectField
                                label=""
                                value={paginationType}
                                onChange={(e) => setPaginationType(e.target.value)}
                                options={[
                                    { value: 'auto', label: strings.pagination_type_auto },
                                    { value: 'url_param', label: strings.pagination_type_url_param },
                                    { value: 'next_button', label: strings.pagination_type_next_button },
                                ]}
                            />
                            {paginationType === 'url_param' && (
                                <InputField
                                    label={strings.page_param_name_label}
                                    value={pageParamName}
                                    onChange={(e) => setPageParamName(e.target.value)}
                                    placeholder={strings.page_param_name_placeholder}
                                />
                            )}
                            {paginationType === 'next_button' && (
                                <InputField
                                    label={strings.next_button_selector_label}
                                    value={nextButtonSelector}
                                    onChange={(e) => setNextButtonSelector(e.target.value)}
                                    placeholder={strings.next_button_selector_placeholder}
                                />
                            )}
                        </div>
                    </div>
                )}

                {/* Concurrency Settings */}
                <div className="mt-8 pt-6 border-t border-gray-700">
                    <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
                        <i data-lucide="cpu" className="w-5 h-5 inline-block mr-2"></i>
                        {strings.concurrency_settings_title}
                    </h3>
                    <CheckboxToggle
                        label={strings.auto_adjust_concurrency_label}
                        checked={autoAdjustConcurrency}
                        onChange={(e) => setAutoAdjustConcurrency(e.target.checked)}
                    >
                        <Tooltip content="Automatically adjust the number of concurrent browser instances based on system resources and scraping performance." />
                    </CheckboxToggle>

                    {autoAdjustConcurrency && (
                        <SelectField
                            label={strings.auto_adjust_mode_label}
                            value={autoAdjustMode}
                            onChange={(e) => setAutoAdjustMode(e.target.value)}
                            options={[
                                { value: 'full_auto', label: strings.auto_adjust_mode_full_auto },
                                { value: 'fixed', label: strings.auto_adjust_mode_fixed },
                            ]}
                        />
                    )}

                    {!autoAdjustConcurrency || autoAdjustMode === 'fixed' ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField
                                label={strings.max_concurrent_browsers_min_label}
                                type="number"
                                value={maxConcurrentBrowsersMin}
                                onChange={(e) => setMaxConcurrentBrowsersMin(e.target.value)}
                                disabled={autoAdjustConcurrency && autoAdjustMode === 'full_auto'}
                            />
                            <InputField
                                label={strings.max_concurrent_browsers_max_label}
                                type="number"
                                value={maxConcurrentBrowsersMax}
                                onChange={(e) => setMaxConcurrentBrowsersMax(e.target.value)}
                                disabled={autoAdjustConcurrency && autoAdjustMode === 'full_auto'}
                            />
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField
                                label={strings.max_concurrent_browsers_min_label}
                                type="number"
                                value={maxConcurrentBrowsersMin}
                                onChange={(e) => setMaxConcurrentBrowsersMin(e.target.value)}
                                disabled={false} // Enable for full auto to set bounds
                            />
                            <InputField
                                label={strings.max_concurrent_browsers_max_label}
                                type="number"
                                value={maxConcurrentBrowsersMax}
                                onChange={(e) => setMaxConcurrentBrowsersMax(e.target.value)}
                                disabled={false} // Enable for full auto to set bounds
                            />
                        </div>
                    )}

                    <CheckboxToggle
                        label={strings.use_random_user_agents_label}
                        checked={useRandomUserAgents}
                        onChange={(e) => setUseRandomUserAgents(e.target.checked)}
                    >
                        <Tooltip content="Rotate through a list of common user agents to mimic different browsers and reduce detection." />
                    </CheckboxToggle>

                    <InputField
                        label={strings.max_empty_category_attempts_label}
                        type="number"
                        value={maxEmptyCategoryAttempts}
                        onChange={(e) => setMaxEmptyCategoryAttempts(e.target.value)}
                    >
                        <Tooltip content="Maximum number of times to retry a category page if no products are found before marking it as 'no products found'." />
                    </InputField>
                </div>

                {/* Image Settings */}
                <div className="mt-8 pt-6 border-t border-gray-700">
                    <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
                        <i data-lucide="image" className="w-5 h-5 inline-block mr-2"></i>
                        Image Management Settings
                    </h3>
                    <CheckboxToggle
                        label={strings.save_images_label}
                        checked={saveImages}
                        onChange={(e) => setSaveImages(e.target.checked)}
                    >
                        <Tooltip content="Download and save product images to the output directory." />
                    </CheckboxToggle>

                    {saveImages && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField
                                label={strings.image_quality_label}
                                type="number"
                                value={imageQuality}
                                onChange={(e) => setImageQuality(e.target.value)}
                                min="1" max="100"
                            >
                                <Tooltip content="Quality of saved JPEG images (1-100). Lower quality means smaller file size." />
                            </InputField>
                            <SelectField
                                label={strings.image_format_label}
                                value={imageFormat}
                                onChange={(e) => setImageFormat(e.target.value)}
                                options={[
                                    { value: 'JPEG', label: strings.image_format_jpeg },
                                    { value: 'PNG', label: strings.image_format_png },
                                ]}
                            >
                                <Tooltip content="Format to save images as. JPEG is generally smaller, PNG supports transparency." />
                            </SelectField>
                        </div>
                    )}
                </div>

                {/* Action Buttons */}
                <div className="mt-8 flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
                    <button
                        onClick={handleStartScraping}
                        className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors shadow-lg transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 flex items-center justify-center"
                    >
                        <i data-lucide="play" className="w-5 h-5 inline-block mr-2"></i>
                        {strings.start_scraping_button}
                    </button>
                    <button
                        onClick={handleStopScraping}
                        className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg transition-colors shadow-lg transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-75 flex items-center justify-center"
                    >
                        <i data-lucide="stop-circle" className="w-5 h-5 inline-block mr-2"></i>
                        {strings.stop_scraping_button}
                    </button>
                </div>
            </section>

            {/* Scraper Status Section */}
            <section className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700">
                <h2 className="text-2xl font-bold text-gray-100 mb-6 flex items-center">
                    <i data-lucide="activity" className="w-6 h-6 inline-block mr-3"></i>
                    {strings.scraper_status_title}
                </h2>
                <StatusDisplay status={scraperStatus} strings={strings} formatTimestamp={formatTimestamp} />
                <h3 className="text-xl font-semibold text-gray-100 mt-6 mb-4 flex items-center">
                    <i data-lucide="clipboard-list" className="w-5 h-5 inline-block mr-2"></i>
                    {strings.recent_logs_title}
                </h3>
                <LogsDisplay logs={recentLogs} strings={strings} formatTimestamp={formatTimestamp} logsEndRef={logsEndRef} />
            </section>

            {/* Data Preview Section */}
            <section className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700 flex flex-col">
                <h2 className="text-2xl font-bold text-gray-100 mb-6 flex items-center">
                    <i data-lucide="eye" className="w-6 h-6 inline-block mr-3"></i>
                    {strings.preview_data_title}
                </h2>

                {/* Scraped Products Preview */}
                <DataPreview
                    title={strings.scraped_products_preview_title}
                    items={scrapedProducts}
                    noItemsMessage={strings.no_scraped_products}
                    renderItem={(product) => (
                        <>
                            <h3 className="text-lg font-semibold text-blue-400 mb-2">{product.name || strings.product_name_unknown}</h3>
                            {product.image_url && (
                                <img
                                    src={product.image_url}
                                    alt={product.name || 'Product Image'}
                                    className="w-full h-32 object-cover rounded-md mb-2 border border-gray-700"
                                    onError={(e) => { e.target.onerror = null; e.target.src = 'https://placehold.co/128x128/334155/E2E8F0?text=No+Image'; }}
                                />
                            )}
                            <p className="text-gray-300 text-sm mb-1"><strong>{strings.price_label}</strong> {product.price || strings.price_unknown}</p>
                            <p className="text-gray-300 text-sm mb-1"><strong>{strings.brand_label}</strong> {product.brand || strings.brand_unknown}</p>
                            <p className="text-gray-300 text-sm mb-1"><strong>{strings.source_label}</strong> <a href={product.product_url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline truncate">{product.product_url}</a></p>
                            {product.description && (
                                <button
                                    onClick={() => handleSummarizeDescription(product.description)}
                                    className="mt-3 bg-purple-600 hover:bg-purple-700 text-white text-xs py-1.5 px-3 rounded-md transition-colors shadow-sm flex items-center justify-center"
                                >
                                    <i data-lucide="sparkles" className="w-4 h-4 inline-block mr-1"></i>
                                    {strings.summarize_description_button}
                                </button>
                            )}
                        </>
                    )}
                />

                {/* Discovered Categories Preview */}
                <div className="mt-8">
                    <DataPreview
                        title={strings.discovered_categories_preview_title}
                        items={discoveredCategories}
                        noItemsMessage={strings.no_discovered_categories}
                        renderItem={(category) => (
                            <>
                                <h3 className="text-lg font-semibold text-blue-400 mb-2">{category.name || strings.category_name_unknown}</h3>
                                <p className="text-gray-300 text-sm mb-1"><strong>URL:</strong> <a href={category.url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline truncate">{category.url}</a></p>
                                <p className="text-gray-300 text-sm mb-1"><strong>Status:</strong> <span className={
                                    category.status === 'pending' ? 'text-yellow-400' :
                                    category.status === 'processing' ? 'text-blue-400' :
                                    category.status === 'processed' ? 'text-green-400' :
                                    category.status === 'no_products_found' ? 'text-red-400' :
                                    'text-gray-300'
                                }>{category.status}</span></p>
                                {category.no_product_attempts > 0 && (
                                    <p className="text-gray-300 text-sm mb-1"><strong>Empty attempts:</strong> {category.no_product_attempts}</p>
                                )}
                            </>
                        )}
                    />
                </div>
            </section>

            {/* Upload/Download Section */}
            <section className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700">
                <h2 className="text-2xl font-bold text-gray-100 mb-6 flex items-center">
                    <i data-lucide="upload" className="w-6 h-6 inline-block mr-3"></i>
                    {strings.upload_categories_title} / <i data-lucide="download" className="w-6 h-6 inline-block mr-3 ml-3"></i> {strings.download_data_title}
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Upload Categories */}
                    <div>
                        <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
                            <i data-lucide="file-up" className="w-5 h-5 inline-block mr-2"></i>
                            {strings.upload_categories_title}
                        </h3>
                        <label className="block text-gray-300 text-sm font-medium mb-2">{strings.choose_file_button}</label>
                        <input
                            type="file"
                            onChange={handleUploadCategories}
                            className="block w-full text-sm text-gray-300
                                file:mr-4 file:py-2 file:px-4
                                file:rounded-md file:border-0
                                file:text-sm file:font-semibold
                                file:bg-blue-600 file:text-white
                                hover:file:bg-blue-700 transition-colors cursor-pointer"
                        />
                        <p className="mt-2 text-gray-400 text-xs">
                            Supported formats: JSON (list of URLs or objects with 'url' key), CSV (with 'url' column).
                        </p>
                    </div>

                    {/* Download Data */}
                    <div>
                        <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
                            <i data-lucide="file-down" className="w-5 h-5 inline-block mr-2"></i>
                            {strings.download_data_title}
                        </h3>
                        <SelectField
                            label={strings.download_type_label}
                            value="products"
                            onChange={(e) => handleDownloadData(e.target.value, document.getElementById('download-format').value)}
                            options={[
                                { value: 'products', label: strings.download_type_products },
                                { value: 'categories', label: strings.download_type_categories },
                                { value: 'logs', label: strings.download_type_logs },
                                { value: 'all', label: strings.download_type_all },
                            ]}
                        />
                        <SelectField
                            label={strings.download_format_label}
                            value="json"
                            onChange={(e) => handleDownloadData(document.getElementById('download-type').value, e.target.value)}
                            options={[
                                { value: 'json', label: strings.download_format_json },
                                { value: 'csv', label: strings.download_format_csv },
                            ]}
                            className="mb-4"
                            id="download-format" // Added ID for easier access
                        />
                        <button
                            onClick={() => handleDownloadData(document.getElementById('download-type').value, document.getElementById('download-format').value)}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md transition-colors shadow-md flex items-center justify-center"
                        >
                            <i data-lucide="download" className="w-5 h-5 inline-block mr-2"></i>
                            {strings.download_button}
                        </button>
                    </div>
                </div>
            </section>

            <Modal
                title={modalTitle}
                content={modalContent}
                onClose={() => setShowModal(false)}
                strings={strings}
            />
        </div>
    );
}

// Render the App component into the 'root' div
ReactDOM.render(<App />, document.getElementById('root'));


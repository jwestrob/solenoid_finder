// Solenoid Viewer App

let resultsData = {};
let currentProtein = null;
let structureViewer = null;
let currentFilter = 'solenoid';
let currentStructureRequest = null;  // Track current fetch to cancel stale requests
let minHarmonics = 4;
let minContinuity = 0;
let minRepeats = 5;
let sortBy = 'votes';

// Initialize the app
async function init() {
    try {
        // Load results data
        const response = await fetch('data/results.json');
        resultsData = await response.json();

        // Initialize the protein list
        renderProteinList();

        // Set up event listeners
        setupEventListeners();

        // Initialize 3Dmol viewer
        initStructureViewer();

        // Select first protein with solenoid
        const firstSolenoid = Object.keys(resultsData).find(id => resultsData[id].has_solenoid);
        if (firstSolenoid) {
            selectProtein(firstSolenoid);
        }
    } catch (error) {
        console.error('Error initializing app:', error);
        document.getElementById('protein-list').innerHTML = '<li class="loading">Error loading data</li>';
    }
}

function setupEventListeners() {
    // Search input
    document.getElementById('search-input').addEventListener('input', (e) => {
        renderProteinList(e.target.value);
    });

    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentFilter = e.target.dataset.filter;
            renderProteinList(document.getElementById('search-input').value);
        });
    });

    // Min harmonics filter
    document.getElementById('min-harmonics').addEventListener('change', (e) => {
        minHarmonics = parseInt(e.target.value) || 1;
        renderProteinList(document.getElementById('search-input').value);
    });

    // Min continuity filter
    document.getElementById('min-continuity').addEventListener('change', (e) => {
        minContinuity = parseFloat(e.target.value) || 0;
        renderProteinList(document.getElementById('search-input').value);
    });

    // Min repeats filter
    document.getElementById('min-repeats').addEventListener('change', (e) => {
        minRepeats = parseFloat(e.target.value) || 1;
        renderProteinList(document.getElementById('search-input').value);
    });

    // Sort dropdown
    document.getElementById('sort-by').addEventListener('change', (e) => {
        sortBy = e.target.value;
        renderProteinList(document.getElementById('search-input').value);
    });
}

function getMaxHarmonics(data) {
    if (!data.regions || data.regions.length === 0) return 0;
    return Math.max(...data.regions.map(r => r.n_harmonics));
}

function getMaxFftProm(data) {
    if (!data.regions || data.regions.length === 0) return -99;
    return Math.max(...data.regions.map(r => r.fft_prominence || 0));
}

function getMaxBandScore(data) {
    if (!data.regions || data.regions.length === 0) return 0;
    return Math.max(...data.regions.map(r => r.band_score || 0));
}

function getMaxContinuity(data) {
    if (!data.regions || data.regions.length === 0) return 0;
    return Math.max(...data.regions.map(r => r.continuity || 0));
}

function getMaxVotes(data) {
    if (!data.regions || data.regions.length === 0) return 0;
    return Math.max(...data.regions.map(r => r.votes || 0));
}

function getMaxRepeats(data) {
    if (!data.regions || data.regions.length === 0) return 0;
    return Math.max(...data.regions.map(r => r.n_repeats || 0));
}

function getMaxRun(data) {
    if (!data.regions || data.regions.length === 0) return 0;
    return Math.max(...data.regions.map(r => r.longest_run_frac || 0));
}

// Check if any region passes the filtering criteria (with rescue rules)
function passesFilter(data, minHarm, minRep, minCont) {
    if (!data.regions || data.regions.length === 0) return false;

    return data.regions.some(r => {
        const nHarm = r.n_harmonics || 0;
        const nRep = r.n_repeats || 0;
        const cont = r.continuity || 0;
        const run = r.longest_run_frac || 0;

        // Must meet minimum repeats
        if (nRep < minRep) return false;

        // Must meet minimum continuity
        if (cont < minCont) return false;

        // Rule 1: Standard threshold (n_harm >= minHarm)
        if (nHarm >= minHarm) return true;

        // Rescue rules only apply when minHarm is at default (4)
        // This allows users to override by setting minHarm to 3 or lower
        if (minHarm <= 3) return nHarm >= minHarm;

        // Rule 2: Continuity rescue (n_harm >= 3 AND cont >= 0.15)
        if (nHarm >= 3 && cont >= 0.15) return true;

        // Rule 3: Run rescue (n_harm >= 3 AND run >= 0.40 AND cont >= 0.10)
        if (nHarm >= 3 && run >= 0.40 && cont >= 0.10) return true;

        return false;
    });
}

function renderProteinList(searchTerm = '') {
    const list = document.getElementById('protein-list');
    const proteins = Object.keys(resultsData)
        .filter(id => {
            const data = resultsData[id];
            const matchesSearch = id.toLowerCase().includes(searchTerm.toLowerCase());
            const matchesFilter = currentFilter === 'all' || data.has_solenoid;
            // Use combined filter with rescue rules
            const passesThresholds = currentFilter === 'all' || passesFilter(data, minHarmonics, minRepeats, minContinuity);
            return matchesSearch && matchesFilter && passesThresholds;
        })
        .sort((a, b) => {
            const aData = resultsData[a];
            const bData = resultsData[b];

            if (sortBy === 'votes') {
                // Sort by max votes descending, then continuity as tiebreaker
                const aVotes = getMaxVotes(aData);
                const bVotes = getMaxVotes(bData);
                if (bVotes !== aVotes) return bVotes - aVotes;
                const aCont = getMaxContinuity(aData);
                const bCont = getMaxContinuity(bData);
                if (bCont !== aCont) return bCont - aCont;
                return a.localeCompare(b);
            } else if (sortBy === 'continuity') {
                // Sort by max continuity descending
                const aCont = getMaxContinuity(aData);
                const bCont = getMaxContinuity(bData);
                if (bCont !== aCont) return bCont - aCont;
                return a.localeCompare(b);
            } else if (sortBy === 'band_score') {
                // Sort by max band_score descending
                const aScore = getMaxBandScore(aData);
                const bScore = getMaxBandScore(bData);
                if (bScore !== aScore) return bScore - aScore;
                return a.localeCompare(b);
            } else if (sortBy === 'n_harmonics') {
                // Sort by max n_harmonics descending
                const aHarm = getMaxHarmonics(aData);
                const bHarm = getMaxHarmonics(bData);
                if (bHarm !== aHarm) return bHarm - aHarm;
                return a.localeCompare(b);
            } else if (sortBy === 'fft_prominence') {
                // Sort by max fft_prominence descending
                const aProm = getMaxFftProm(aData);
                const bProm = getMaxFftProm(bData);
                if (bProm !== aProm) return bProm - aProm;
                return a.localeCompare(b);
            } else if (sortBy === 'length') {
                // Sort by length descending
                if (bData.length !== aData.length) return bData.length - aData.length;
                return a.localeCompare(b);
            } else {
                // Alphabetical - solenoids first
                const aHas = aData.has_solenoid;
                const bHas = bData.has_solenoid;
                if (aHas && !bHas) return -1;
                if (!aHas && bHas) return 1;
                return a.localeCompare(b);
            }
        });

    list.innerHTML = proteins.map(id => {
        const data = resultsData[id];
        const isActive = id === currentProtein;
        const hasSolenoid = data.has_solenoid;
        const maxHarm = getMaxHarmonics(data);
        const maxCont = getMaxContinuity(data);
        const maxVotes = getMaxVotes(data);

        return `
            <li class="protein-item ${isActive ? 'active' : ''} ${hasSolenoid ? 'has-solenoid' : ''}"
                data-id="${id}">
                <span>${id}</span>
                <span class="badges">
                    ${hasSolenoid ? `<span class="badge badge-votes">${maxVotes}/5</span>` : ''}
                    ${hasSolenoid ? `<span class="badge">n=${maxHarm}</span>` : ''}
                    ${hasSolenoid ? `<span class="badge badge-cont">c=${maxCont.toFixed(2)}</span>` : ''}
                </span>
            </li>
        `;
    }).join('');

    // Add click listeners
    list.querySelectorAll('.protein-item').forEach(item => {
        item.addEventListener('click', () => selectProtein(item.dataset.id));
    });
}

async function selectProtein(proteinId) {
    currentProtein = proteinId;
    const data = resultsData[proteinId];

    // Update list selection
    document.querySelectorAll('.protein-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === proteinId);
    });

    // Update info panel
    document.getElementById('info-id').textContent = proteinId;
    document.getElementById('info-length').textContent = data.length;

    // Get metrics from first/best region
    const r = data.regions.length > 0 ? data.regions[0] : null;
    document.getElementById('info-harmonics').textContent = r ? r.n_harmonics : '-';
    document.getElementById('info-repeats').textContent = r ? (r.n_repeats || 0).toFixed(1) : '-';
    document.getElementById('info-continuity').textContent = r ? (r.continuity || 0).toFixed(3) : '-';
    document.getElementById('info-fft').textContent = r ? (r.fft_prominence || 0).toFixed(2) : '-';
    document.getElementById('info-band-score').textContent = r ? (r.band_score || 0).toFixed(3) : '-';

    // Render region details with all metrics including votes
    const regionList = document.getElementById('region-list');
    if (data.regions.length > 0) {
        regionList.innerHTML = data.regions.map((r, i) => {
            const vd = r.vote_details || {};
            const voteStr = `[${vd.harmonics ? 'H' : '-'}${vd.continuity ? 'C' : '-'}${vd.fft ? 'F' : '-'}${vd.band ? 'B' : '-'}${vd.run ? 'R' : '-'}]`;
            return `
                <div class="region-item">
                    Region ${i + 1}: ${r.start}-${r.end} (${r.end - r.start} aa) | period=${r.period || '?'} | n_rep=${(r.n_repeats || 0).toFixed(1)} | <b>votes=${r.votes || 0}/5 ${voteStr}</b>
                    <br>n_harm=${r.n_harmonics} | cont=${(r.continuity || 0).toFixed(3)} | run=${(r.longest_run_frac || 0).toFixed(2)} | fft=${(r.fft_prominence || 0).toFixed(2)} | band=${(r.band_score || 0).toFixed(3)}
                </div>
            `;
        }).join('');
    } else {
        regionList.innerHTML = '';
    }

    // Load APC image
    loadAPCImage(proteinId, data);

    // Load structure
    loadStructure(proteinId, data);
}

function loadAPCImage(proteinId, data) {
    const container = document.getElementById('apc-viewer');

    if (data.has_solenoid) {
        // Load pre-rendered image
        const img = new Image();
        img.onload = () => {
            container.innerHTML = '';
            container.appendChild(img);
            img.style.maxWidth = '100%';
            img.style.maxHeight = '100%';
            img.style.objectFit = 'contain';
            img.style.display = 'block';
            img.style.margin = 'auto';
        };
        img.onerror = () => {
            container.innerHTML = '<div class="loading">APC image not available</div>';
        };
        img.src = `apc_images/${proteinId}.png`;
    } else {
        container.innerHTML = '<div class="loading">No solenoid detected - APC image not generated</div>';
    }
}

function initStructureViewer() {
    const element = document.getElementById('structure-viewer');
    structureViewer = $3Dmol.createViewer(element, {
        backgroundColor: '#1a1a2e'
    });
    // Ensure viewer resizes properly
    structureViewer.resize();
    window.addEventListener('resize', () => {
        if (structureViewer) structureViewer.resize();
    });
}

async function loadStructure(proteinId, data) {
    if (!structureViewer) return;

    // Cancel any pending request
    if (currentStructureRequest) {
        currentStructureRequest.abort();
    }
    const abortController = new AbortController();
    currentStructureRequest = abortController;
    const requestId = proteinId;  // Track which protein this request is for

    structureViewer.clear();

    const container = document.getElementById('structure-viewer');

    // Remove any existing loading divs
    container.querySelectorAll('.loading').forEach(el => el.remove());

    // Show loading state
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.textContent = 'Loading structure...';
    loadingDiv.style.position = 'absolute';
    loadingDiv.style.top = '50%';
    loadingDiv.style.left = '50%';
    loadingDiv.style.transform = 'translate(-50%, -50%)';
    loadingDiv.style.zIndex = '10';
    container.appendChild(loadingDiv);

    try {
        // First get the correct URL from AlphaFold API
        const apiUrl = `https://alphafold.ebi.ac.uk/api/prediction/${proteinId}`;
        const apiResponse = await fetch(apiUrl, { signal: abortController.signal });

        if (!apiResponse.ok) throw new Error('Protein not in AlphaFold');
        const apiData = await apiResponse.json();

        const pdbUrl = apiData[0]?.pdbUrl;
        if (!pdbUrl) throw new Error('No PDB URL found');

        // Check if we're still the current request
        if (currentProtein !== requestId) return;

        const pdbResponse = await fetch(pdbUrl, { signal: abortController.signal });
        if (!pdbResponse.ok) throw new Error('Structure not found');
        const pdbData = await pdbResponse.text();

        // Final check - make sure we're still the current protein
        if (currentProtein !== requestId) return;

        // Remove loading indicator
        loadingDiv.remove();

        // Add model to viewer
        structureViewer.addModel(pdbData, 'pdb');

        // Color by solenoid regions using range selector (more efficient)
        if (data.regions.length > 0) {
            // Default color (gray)
            structureViewer.setStyle({}, {
                cartoon: { color: '#666666' }
            });

            // Highlight solenoid regions (red) - use range strings instead of arrays
            data.regions.forEach(region => {
                const start = region.start + 1;  // PDB is 1-indexed
                const end = region.end;
                structureViewer.setStyle(
                    { resi: [start + '-' + end] },
                    { cartoon: { color: '#e94560' } }
                );
            });
        } else {
            // No solenoid - color by spectrum
            structureViewer.setStyle({}, {
                cartoon: { color: 'spectrum' }
            });
        }

        structureViewer.zoomTo();
        structureViewer.resize();
        structureViewer.render();

    } catch (error) {
        // Ignore abort errors (user switched proteins)
        if (error.name === 'AbortError') return;

        loadingDiv.textContent = 'Structure not available from AlphaFold';
        console.error('Error loading structure:', error);
    }
}


// Start the app
init();

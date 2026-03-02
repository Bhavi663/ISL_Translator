// Accessibility Module
class AccessibilityManager {
    constructor() {
        this.settings = {
            darkMode: false,
            largeText: false,
            highContrast: false,
            reducedMotion: false
        };
        
        this.loadSettings();
        this.applySettings();
        this.setupEventListeners();
    }
    
    loadSettings() {
        // Load from localStorage
        const saved = localStorage.getItem('accessibilitySettings');
        if (saved) {
            try {
                this.settings = JSON.parse(saved);
            } catch (e) {
                console.warn('Failed to load accessibility settings');
            }
        }
    }
    
    saveSettings() {
        localStorage.setItem('accessibilitySettings', JSON.stringify(this.settings));
    }
    
    applySettings() {
        const body = document.body;
        
        // Dark Mode
        if (this.settings.darkMode) {
            body.classList.add('dark-mode');
        } else {
            body.classList.remove('dark-mode');
        }
        
        // Large Text
        if (this.settings.largeText) {
            body.classList.add('large-text');
        } else {
            body.classList.remove('large-text');
        }
        
        // High Contrast
        if (this.settings.highContrast) {
            body.classList.add('high-contrast');
        } else {
            body.classList.remove('high-contrast');
        }
        
        // Reduced Motion
        if (this.settings.reducedMotion) {
            body.classList.add('reduced-motion');
        } else {
            body.classList.remove('reduced-motion');
        }
        
        // Update toggle buttons if they exist
        this.updateToggleButtons();
    }
    
    toggle(setting) {
        this.settings[setting] = !this.settings[setting];
        this.saveSettings();
        this.applySettings();
        
        // Announce change for screen readers
        this.announceChange(setting, this.settings[setting]);
    }
    
    announceChange(setting, enabled) {
        const message = `${setting} mode ${enabled ? 'enabled' : 'disabled'}`;
        const announcer = document.getElementById('accessibilityAnnouncer');
        if (announcer) {
            announcer.textContent = message;
        } else {
            const div = document.createElement('div');
            div.id = 'accessibilityAnnouncer';
            div.setAttribute('aria-live', 'polite');
            div.className = 'sr-only';
            document.body.appendChild(div);
            setTimeout(() => {
                div.textContent = message;
            }, 100);
        }
    }// Add to AccessibilityManager class
async syncWithServer() {
    try {
        const response = await fetch('/api/accessibility/load');
        const data = await response.json();
        
        // Merge server settings with local
        this.settings = { ...this.settings, ...data };
        this.saveSettings();
        this.applySettings();
    } catch (e) {
        console.warn('Failed to sync with server', e);
    }
}

async saveToServer() {
    try {
        await fetch('/api/accessibility/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(this.settings)
        });
    } catch (e) {
        console.warn('Failed to save to server', e);
    }
}


// Override toggle to sync with server
toggle(setting) {
    this.settings[setting] = !this.settings[setting];
    this.saveSettings();
    this.applySettings();
    this.saveToServer(); // Sync with server
    this.announceChange(setting, this.settings[setting]);
}

// Initialize with server sync
async init() {
    await this.syncWithServer();
    this.setupEventListeners();
}
    
    setupEventListeners() {
        // Listen for system dark mode preference
        const darkModeMedia = window.matchMedia('(prefers-color-scheme: dark)');
        darkModeMedia.addEventListener('change', (e) => {
            if (!localStorage.getItem('accessibilitySettings')) {
                this.settings.darkMode = e.matches;
                this.applySettings();
            }
        });
        
        // Listen for system reduced motion preference
        const motionMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
        motionMedia.addEventListener('change', (e) => {
            this.settings.reducedMotion = e.matches;
            this.applySettings();
        });
    }
    
    updateToggleButtons() {
        // Update all accessibility toggle buttons
        document.querySelectorAll('[data-accessibility-toggle]').forEach(btn => {
            const setting = btn.dataset.accessibilityToggle;
            if (this.settings[setting]) {
                btn.classList.add('active');
                btn.setAttribute('aria-pressed', 'true');
            } else {
                btn.classList.remove('active');
                btn.setAttribute('aria-pressed', 'false');
            }
        });
    }
    
    getCurrentSettings() {
        return { ...this.settings };
    }
}

// Initialize accessibility manager
const accessibility = new AccessibilityManager();

// Make globally available
window.accessibility = accessibility;
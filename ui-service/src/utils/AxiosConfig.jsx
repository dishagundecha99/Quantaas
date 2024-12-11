import axios from 'axios';

const axiosConfig = axios.create({
    baseURL: window._env_.API_URL,
    Headers: {
        'Content-Type': 'application/json',
    }
});

export default axiosConfig;
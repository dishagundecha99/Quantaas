import AddIcon from '@mui/icons-material/Add';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import LoopIcon from '@mui/icons-material/Loop';
import RefreshIcon from '@mui/icons-material/Refresh';
import SearchIcon from '@mui/icons-material/Search';
import { Button, Grid, IconButton, InputAdornment, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField } from '@mui/material';
import React, { useEffect, useState } from 'react';
import AddJobModal from '../components/AddJobModal';
import axiosConfig from '../utils/AxiosConfig';

const RunningStats = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [tableData, setTableData] = useState([]);
    const [openAddModal, setOpenAddModal] = useState(false);

    const user_id = localStorage.getItem("user_id");

    const fetchRunningJobs = async () => {

        try {
            const response = (await axiosConfig.get('/current-run', { params: { "user-id": user_id } })).data;
            console.log('Response:', response);
              setTableData(response?.["current-run"]?.map((data) => (
                {
                  exp_id: data.exp_id,
                  name: (data.exp_id).substr(0, data.exp_id?.lastIndexOf('_')),
                  learning_rate: data?.hyperparams?.learning_rate,
                  weight_decay: data?.hyperparams?.weight_decay,
                  batch_size: data?.hyperparams?.batch_size,
                  max_epochs: data?.hyperparams?.max_epochs,
                  warmup_steps: data?.hyperparams?.warmup_steps,
                  training: data?.training,
                }
              )) || []);
        } catch (error) {
            console.error('Error fetching trade details:', error);
        }
    };

    useEffect(() => {
        fetchRunningJobs();
    }, []);

    const handleSearch = (event) => {
        setSearchTerm(event.target.value);
    };

    const filteredData = tableData.filter((data) =>
        // TODO: change based on experiment name
        data.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleOpenDialog = () => {
        setOpenAddModal(true);
    };

    const handleCloseDialog = () => {
        setOpenAddModal(false);
    };

    return (
        <div>
            <Grid container spacing={2} pt={2} pb={2} sx={{ height: "50%" }}>
                <Grid item xs={6} sm={6} >
                    <TextField
                        label="Search"
                        variant="outlined"
                        size='small'
                        value={searchTerm}
                        sx={{ width: '100%' }}
                        InputProps={{
                            endAdornment: (
                                <InputAdornment>
                                    <IconButton>
                                        <SearchIcon />
                                    </IconButton>
                                </InputAdornment>
                            )
                        }}
                        onChange={handleSearch}
                    />
                </Grid>
                <Grid container item xs={6} sm={6} justifyContent={"flex-end"}>
                    <Button variant="outlined" color="secondary" startIcon={<AddIcon />} onClick={handleOpenDialog}>
                        Add Job
                    </Button>
                    <IconButton onClick={fetchRunningJobs}>
                        <RefreshIcon />
                    </IconButton>
                </Grid>
            </Grid>
            <Grid container spacing={2}>
                <Grid item xs={12} sm={12}>
                    <TableContainer component={Paper}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>Experiment Name</TableCell>
                                    <TableCell>Experiment ID</TableCell>
                                    <TableCell>Learning Rate</TableCell>
                                    <TableCell>Weight Decay</TableCell>
                                    <TableCell>Batch Size</TableCell>
                                    <TableCell>Max Epochs</TableCell>
                                    <TableCell>Warmup Steps</TableCell>
                                    <TableCell>Status</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {filteredData.length === 0 ?
                                    <TableRow>
                                        <TableCell colSpan={8} sx={{ bgcolor: 'lightgray', fontStyle: "italic" }} align="center">No data found</TableCell>
                                    </TableRow>
                                    :
                                    filteredData.map((data) => (
                                        <TableRow key={data.exp_id}>
                                            <TableCell>{data.name}</TableCell>
                                            <TableCell>{data.exp_id}</TableCell>
                                            <TableCell>{data.learning_rate}</TableCell>
                                            <TableCell>{data.weight_decay}</TableCell>
                                            <TableCell>{data.batch_size}</TableCell>
                                            <TableCell>{data.max_epochs}</TableCell>
                                            <TableCell>{data.warmup_steps}</TableCell>
                                            <TableCell>
                                                {data.training ? (
                                                    <LoopIcon
                                                        sx={{
                                                            animation: "spin 2s linear infinite",
                                                            "@keyframes spin": {
                                                                "0%": {
                                                                    transform: "rotate(360deg)",
                                                                },
                                                                "100%": {
                                                                    transform: "rotate(0deg)",
                                                                },
                                                            },
                                                        }}
                                                    />
                                                ) : (
                                                    <CheckCircleOutlineIcon sx={{ color: "green" }} />
                                                )
                                                }
                                            </TableCell>
                                        </TableRow>
                                    ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Grid>
            </Grid>

            {/* Add Job Modal */}
            <AddJobModal open={openAddModal} handleClose={handleCloseDialog} />
        </div>
    );
};
export default RunningStats;
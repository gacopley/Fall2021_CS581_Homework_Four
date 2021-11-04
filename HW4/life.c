/* 
   Revamped Game of Life Program Utilizing MPI
   Author: Alex Copley
   Date: November 3rd, 2021

   To compile: mpicc -g -Wall -o life life.c
   To run: mpiexec [-n numProcesses] ./life -n <problem size> -max <max iterations> -fo <output dir>
           mpiexec 4 ./life -n 1000 -max 1000 -fo .                   (on your local system)
           mpiexec 4 ./life -n 1000 -max 1000 -fo /scratch/$USER/     (on DMC at ASC)
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

#define TAG 0
char **next_gen;
int updated;
char **cur_gen;
int local_N;
int local_M;

//Allocating memory to the "board's" array.
char **allocate_memory(int rows, int columns)
{
    int i;
    char *data = malloc(rows * columns * sizeof(char));
    char **arr = malloc(rows * sizeof(char *));
    for (i = 0; i < rows; i++)
        arr[i] = &(data[i * columns]);

    return arr;
}

void calculate_rows_columns(int *num_rows, int *num_cols, int size, int ROWS, int COLS, int NPROWS, int NPCOLS)
{
    int i, j;
    for (i = 0; i < size; i++)
    {
        num_rows[i] = ROWS / NPROWS;
        num_cols[i] = COLS / NPCOLS;
    }
    for (i = 0; i < (ROWS % NPROWS); i++)
    {
        for (j = 0; j < NPCOLS; j++)
        {
            num_rows[i * NPCOLS + j]++;
        }
    }
    for (i = 0; i < (COLS % NPCOLS); i++)
    {
        for (j = 0; j < NPROWS; j++)
        {
            num_cols[i + NPROWS * j]++;
        }
    }
}

void create_datatype(MPI_Datatype *derivedtype, int start1, int start2, int subsize1, int subsize2)
{
    const int array_of_bigsizes[2] = {local_N + 2, local_M + 2};
    const int array_of_subsizes[2] = {subsize1, subsize2};
    const int array_of_starts[2] = {start1, start2};

    MPI_Type_create_subarray(2, array_of_bigsizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_CHAR, derivedtype);
    MPI_Type_commit(derivedtype);
}

void determine_status(int i, int j, int sum)
{

    //If cell is alive
    if (cur_gen[i][j] == '1')
    {
        if (sum == 0 || sum == 1)
        {
            next_gen[i][j] = '0';
            updated = 1;
        }
        else if (sum == 2 || sum == 3)
            next_gen[i][j] = '1';

        else if (sum >= 4 && sum <= 8)
        {
            next_gen[i][j] = '0';
            updated = 1;
        }
    }
    //If cell is not alive and has 3 active neighbors
    else if (sum == 3)
    {
        next_gen[i][j] = '1';
        updated = 1;
    }
    //If cell is not alive but it has less than 3 active neighbors, then it remains dead
    else
        next_gen[i][j] = '0';
}

void check_neighbors(int current_i, int current_j, int *sum)
{
    int i, j;

    *sum = 0;

    //For all 8 neighbors
    for (i = -1; i <= 1; ++i)
    {
        for (j = -1; j <= 1; ++j)
        {
            if (i || j)
                //If neighbor is alive,add it to sum
                if (cur_gen[current_i + i][current_j + j] == '1')
                    (*sum)++;
        }
    }
}

void calculate_outer_array(void)
{
    int i, j;
    int sum;
    //For all the border-cells
    for (i = 1; i <= local_N; ++i)
        for (j = 1; j <= local_M; ++j)
        {
            if (i == 1 || i == local_N || j == 1 || j == local_M)
            {
                check_neighbors(i, j, &sum);
                determine_status(i, j, sum);
            }
        }
}

void calculate_inner_array(void)
{
    int i, j;
    int sum;

    //For all cells that require no communication at all
    for (i = 2; i <= local_N - 1; ++i)
        for (j = 2; j <= local_M - 1; ++j)
        {
            check_neighbors(i, j, &sum);
            determine_status(i, j, sum);
        }
}

void find_neighbors(MPI_Comm comm_2D, int my_rank, int NPROWS, int NPCOLS, int *left, int *right, int *top, int *bottom, int *topleft, int *topright, int *bottomleft, int *bottomright)
{

    int disp = 1;
    int my_coords[2];
    int corner_coords[2];

    //Finding top/bottom neighbors
    MPI_Cart_shift(comm_2D, 0, disp, top, bottom);

    //Finding left/right neighbors
    MPI_Cart_shift(comm_2D, 1, disp, left, right);

    //Finding top-right corner
    MPI_Cart_coords(comm_2D, my_rank, 2, my_coords);
    corner_coords[0] = my_coords[0] - 1;
    corner_coords[1] = (my_coords[1] + 1) % NPCOLS;
    if (corner_coords[0] < 0)
        corner_coords[0] = NPROWS - 1;
    MPI_Cart_rank(comm_2D, corner_coords, topright);

    //Finding top-left corner
    MPI_Cart_coords(comm_2D, my_rank, 2, my_coords);
    corner_coords[0] = my_coords[0] - 1;
    corner_coords[1] = my_coords[1] - 1;
    if (corner_coords[0] < 0)
        corner_coords[0] = NPROWS - 1;
    if (corner_coords[1] < 0)
        corner_coords[1] = NPCOLS - 1;
    MPI_Cart_rank(comm_2D, corner_coords, topleft);

    //Finding bottom-right corner
    MPI_Cart_coords(comm_2D, my_rank, 2, my_coords);
    corner_coords[0] = (my_coords[0] + 1) % NPROWS;
    corner_coords[1] = (my_coords[1] + 1) % NPCOLS;
    MPI_Cart_rank(comm_2D, corner_coords, bottomright);

    //Finding bottom-left corner
    MPI_Cart_coords(comm_2D, my_rank, 2, my_coords);
    corner_coords[0] = (my_coords[0] + 1) % NPROWS;
    corner_coords[1] = my_coords[1] - 1;
    if (corner_coords[1] < 0)
        corner_coords[1] = NPCOLS - 1;
    MPI_Cart_rank(comm_2D, corner_coords, bottomleft);
}

int repeating(MPI_Comm comm_2D, int my_rank)
{
    int sum;
    MPI_Allreduce(&updated, &sum, 1, MPI_INT, MPI_SUM, comm_2D);

    if (sum == 0)
        return 1;
    else
        return 0;
}

void gameblocking(MPI_Comm comm_2D, int my_rank, int NPROWS, int NPCOLS, int MAX_GENS)
{

    int gen;
    char **temp;
    next_gen = allocate_memory(local_N + 2, local_M + 2);

    //8 statuses: don't need seperate send & receive commands.
    MPI_Status array_of_statuses[8];

    //Create 4 datatypes for sending
    MPI_Datatype firstcolumn_send, firstrow_send, lastcolumn_send, lastrow_send;
    create_datatype(&firstcolumn_send, 1, 1, local_N, 1);
    create_datatype(&firstrow_send, 1, 1, 1, local_M);
    create_datatype(&lastcolumn_send, 1, local_M, local_N, 1);
    create_datatype(&lastrow_send, local_N, 1, 1, local_M);

    //Create 4 datatypes for receiving
    MPI_Datatype firstcolumn_recv, firstrow_recv, lastcolumn_recv, lastrow_recv;
    create_datatype(&firstcolumn_recv, 1, 0, local_N, 1);
    create_datatype(&firstrow_recv, 0, 1, 1, local_M);
    create_datatype(&lastcolumn_recv, 1, local_M + 1, local_N, 1);
    create_datatype(&lastrow_recv, local_N + 1, 1, 1, local_M);

    //Find ranks of my 8 neighbors
    int left, right, bottom, top, topleft, topright, bottomleft, bottomright;
    find_neighbors(comm_2D, my_rank, NPROWS, NPCOLS, &left, &right, &top, &bottom, &topleft, &topright, &bottomleft, &bottomright);

    for (gen = 0; gen < MAX_GENS; gen++)
    {
        updated = 0;

        //Start all requests [8 sends + 8 receives]
        MPI_Sendrecv(&(cur_gen[0][0]), 1, firstcolumn_send, left, TAG, &(cur_gen[0][0]), 1, firstcolumn_recv, left, TAG, comm_2D, &array_of_statuses[0]);
        MPI_Sendrecv(&(cur_gen[0][0]), 1, firstrow_send, top, TAG, &(cur_gen[0][0]), 1, firstrow_recv, top, TAG, comm_2D, &array_of_statuses[1]);
        MPI_Sendrecv(&(cur_gen[0][0]), 1, lastcolumn_send, right, TAG, &(cur_gen[0][0]), 1, lastcolumn_recv, right, TAG, comm_2D, &array_of_statuses[2]);
        MPI_Sendrecv(&(cur_gen[0][0]), 1, lastrow_send, bottom, TAG, &(cur_gen[0][0]), 1, lastrow_recv, bottom, TAG, comm_2D, &array_of_statuses[3]);
        MPI_Sendrecv(&(cur_gen[1][1]), 1, MPI_CHAR, topleft, TAG, &(cur_gen[0][0]), 1, MPI_CHAR, topleft, TAG, comm_2D, &array_of_statuses[4]);
        MPI_Sendrecv(&(cur_gen[1][local_M]), 1, MPI_CHAR, topright, TAG, &(cur_gen[0][local_M + 1]), 1, MPI_CHAR, topright, TAG, comm_2D, &array_of_statuses[5]);
        MPI_Sendrecv(&(cur_gen[local_N][local_M]), 1, MPI_CHAR, bottomright, TAG, &(cur_gen[local_N + 1][local_M + 1]), 1, MPI_CHAR, bottomright, TAG, comm_2D, &array_of_statuses[6]);
        MPI_Sendrecv(&(cur_gen[local_N][1]), 1, MPI_CHAR, bottomleft, TAG, &(cur_gen[local_N + 1][0]), 1, MPI_CHAR, bottomleft, TAG, comm_2D, &array_of_statuses[7]);

        //Overlap communication [calculating inner matrix]
        calculate_inner_array();

        //We are ready to calculate the outer matrix
        calculate_outer_array();

        //Check if it has remained the same using a flag
        if (repeating(comm_2D, my_rank))
        {
            if (!my_rank)
                printf("No change on %d generation\n", gen);
            break;
        }

        //next_gen will become our local matrix[=our current gen]
        temp = cur_gen;
        cur_gen = next_gen;
        next_gen = temp;
    }

    //Free resources
    MPI_Type_free(&firstcolumn_send);
    MPI_Type_free(&firstrow_send);
    MPI_Type_free(&lastcolumn_send);
    MPI_Type_free(&lastrow_send);

    MPI_Type_free(&firstcolumn_recv);
    MPI_Type_free(&firstrow_recv);
    MPI_Type_free(&lastcolumn_recv);
    MPI_Type_free(&lastrow_recv);

    free(next_gen[0]);
    free(next_gen);
}

void gamenonblocking(MPI_Comm comm_2D, int my_rank, int NPROWS, int NPCOLS, int MAX_GENS)
{

    int gen;
    char **temp;
    next_gen = allocate_memory(local_N + 2, local_M + 2);

    //16 requests , 16 statuses
    MPI_Request array_of_requests[16];
    MPI_Status array_of_statuses[16];

    //Create 4 datatypes for sending
    MPI_Datatype firstcolumn_send, firstrow_send, lastcolumn_send, lastrow_send;
    create_datatype(&firstcolumn_send, 1, 1, local_N, 1);
    create_datatype(&firstrow_send, 1, 1, 1, local_M);
    create_datatype(&lastcolumn_send, 1, local_M, local_N, 1);
    create_datatype(&lastrow_send, local_N, 1, 1, local_M);

    //Create 4 datatypes for receiving
    MPI_Datatype firstcolumn_recv, firstrow_recv, lastcolumn_recv, lastrow_recv;
    create_datatype(&firstcolumn_recv, 1, 0, local_N, 1);
    create_datatype(&firstrow_recv, 0, 1, 1, local_M);
    create_datatype(&lastcolumn_recv, 1, local_M + 1, local_N, 1);
    create_datatype(&lastrow_recv, local_N + 1, 1, 1, local_M);

    //Find ranks of my 8 neighbors
    int left, right, bottom, top, topleft, topright, bottomleft, bottomright;
    find_neighbors(comm_2D, my_rank, NPROWS, NPCOLS, &left, &right, &top, &bottom, &topleft, &topright, &bottomleft, &bottomright);

    MPI_Send_init(&(cur_gen[0][0]), 1, firstcolumn_send, left, TAG, comm_2D, &array_of_requests[0]);
    MPI_Send_init(&(cur_gen[0][0]), 1, firstrow_send, top, TAG, comm_2D, &array_of_requests[1]);
    MPI_Send_init(&(cur_gen[0][0]), 1, lastcolumn_send, right, TAG, comm_2D, &array_of_requests[2]);
    MPI_Send_init(&(cur_gen[0][0]), 1, lastrow_send, bottom, TAG, comm_2D, &array_of_requests[3]);
    MPI_Send_init(&(cur_gen[1][1]), 1, MPI_CHAR, topleft, TAG, comm_2D, &array_of_requests[4]);
    MPI_Send_init(&(cur_gen[1][local_M]), 1, MPI_CHAR, topright, TAG, comm_2D, &array_of_requests[5]);
    MPI_Send_init(&(cur_gen[local_N][local_M]), 1, MPI_CHAR, bottomright, TAG, comm_2D, &array_of_requests[6]);
    MPI_Send_init(&(cur_gen[local_N][1]), 1, MPI_CHAR, bottomleft, TAG, comm_2D, &array_of_requests[7]);

    MPI_Recv_init(&(cur_gen[0][0]), 1, firstcolumn_recv, left, TAG, comm_2D, &array_of_requests[8]);
    MPI_Recv_init(&(cur_gen[0][0]), 1, firstrow_recv, top, TAG, comm_2D, &array_of_requests[9]);
    MPI_Recv_init(&(cur_gen[0][0]), 1, lastcolumn_recv, right, TAG, comm_2D, &array_of_requests[10]);
    MPI_Recv_init(&(cur_gen[0][0]), 1, lastrow_recv, bottom, TAG, comm_2D, &array_of_requests[11]);
    MPI_Recv_init(&(cur_gen[0][0]), 1, MPI_CHAR, topleft, TAG, comm_2D, &array_of_requests[12]);
    MPI_Recv_init(&(cur_gen[0][local_M + 1]), 1, MPI_CHAR, topright, TAG, comm_2D, &array_of_requests[13]);
    MPI_Recv_init(&(cur_gen[local_N + 1][local_M + 1]), 1, MPI_CHAR, bottomright, TAG, comm_2D, &array_of_requests[14]);
    MPI_Recv_init(&(cur_gen[local_N + 1][0]), 1, MPI_CHAR, bottomleft, TAG, comm_2D, &array_of_requests[15]);

    for (gen = 0; gen < MAX_GENS; gen++)
    {
        updated = 0;

        //Start all requests [8 sends + 8 receives]
        MPI_Startall(16, array_of_requests);

        //Overlap communication [calculating inner matrix]
        calculate_inner_array();

        //Make sure all requests are completed
        MPI_Waitall(16, array_of_requests, array_of_statuses);

        //We are ready to calculate the outer matrix
        calculate_outer_array();

        //Check if it has remained the same using a flag
        if (repeating(comm_2D, my_rank))
        {
            if (!my_rank)
                printf("No change on %d generation\n", gen);
            break;
        }

        //next_gen will become new local matrix
        temp = cur_gen;
        cur_gen = next_gen;
        next_gen = temp;
    }

    //Free resources
    MPI_Type_free(&firstcolumn_send);
    MPI_Type_free(&firstrow_send);
    MPI_Type_free(&lastcolumn_send);
    MPI_Type_free(&lastrow_send);

    MPI_Type_free(&firstcolumn_recv);
    MPI_Type_free(&firstrow_recv);
    MPI_Type_free(&lastcolumn_recv);
    MPI_Type_free(&lastrow_recv);

    free(next_gen[0]);
    free(next_gen);
}

int main(int argc, char **argv)
{
    int size, rank, i, j, COLS, ROWS, MAX_GENS;
    int flag1 = 0, flag2 = 0, flag3 = 0;
    double local_start, local_finish, local_elapsed, elapsed;
    FILE *fp;
    char filename[BUFSIZ];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    for (i = 0; i < argc; ++i)
    {
        if (!strcmp("-n", argv[i]))
        {
            ROWS = atoi(argv[i + 1]) * 2;
            COLS = (atoi(argv[i + 1]) * 2);
            flag1 = 1;
        }
        else if (!strcmp("-max", argv[i]))
        {

            MAX_GENS = atoi(argv[i + 1]);
            flag2 = 1;
        }

        else if (!strcmp("-fo", argv[i]))
        {
            sprintf(filename, "%s/output.%d.%d", argv[i + 1], ROWS / 2, MAX_GENS);
            flag3 = 1;
        }
    }

    //Make sure the program is receiving all the information it needs.
    if (!flag1 || !flag2 || !flag3)
    {
        if (rank == 0)
            printf("Usage:mpiexec [-n <NoPROCESSES>] ./life -n <ROWS> -max <MAX_GENS> -fo <outputfile [-fi <inputfile>]\nExiting...\n\n");
        MPI_Finalize();
        exit(1);
    }
    // Setup virtual 2D topology
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {1, 1}; //Periodicity in both dimensions
    int my_coords[2];
    MPI_Comm comm_2D;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2D);
    MPI_Cart_coords(comm_2D, rank, 2, my_coords);

    const int NPROWS = dims[0]; // Number of 'block' rows
    const int NPCOLS = dims[1]; // Number of 'block' cols
    int *num_rows;              // Number of rows for the i-th process [local_N]
    int *num_cols;              // Number of columns for the i-th process [local_M]

    MPI_Barrier(comm_2D);

    // Calculate rows,cols,displacement,extent for each process
    if (rank == 0)
    {

        num_rows = (int *)malloc(size * sizeof(int));
        num_cols = (int *)malloc(size * sizeof(int));

        calculate_rows_columns(num_rows, num_cols, size, ROWS, COLS, NPROWS, NPCOLS);
    }

    // Scatter dimensions,displacement,extent of each process
    MPI_Scatter(num_rows, 1, MPI_INT, &local_N, 1, MPI_INT, 0, comm_2D);
    MPI_Scatter(num_cols, 1, MPI_INT, &local_M, 1, MPI_INT, 0, comm_2D);

    if (rank == 0)
    {
        free(num_rows);
        free(num_cols);
    }

    cur_gen = allocate_memory(local_N + 2, local_M + 2);

    for (i = 1; i <= local_N; ++i)
        for (j = 1; j <= local_M; ++j)
            if (rand() % 2)
                cur_gen[i][j] = '1';
            else
                cur_gen[i][j] = '0';

    /*Each process will start the game
     *gameblocking uses MPI_Sendrecv, while game nonblocking uses MPI_Send and MPI_Recv
     *enabling both simulataneously is not tested, only enable one at a time!*/
    gameblocking(comm_2D, rank, NPROWS, NPCOLS, MAX_GENS);
    //gamenonblocking(comm_2D, rank, NPROWS, NPCOLS, MAX_GENS);

    if (rank == 0)
    {
        fp = fopen(filename, "w+");
        int i, j;
        for (i = 1; i <= local_N; ++i)
        {
            for (j = 1; j <= local_M; ++j)
                fprintf(fp, "%c", cur_gen[i][j]);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    free(cur_gen[0]);
    free(cur_gen);

    MPI_Finalize();
}

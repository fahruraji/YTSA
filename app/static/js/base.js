$(document).ready(function() {
    // let $select = $('select'), 
    // $parent = $select.parent();

    // $select.select2({
    //     dropdownParent: $parent
    // });


    $("#overlay").hide();
    $("form").submit(function () {
        $("#overlay").show();
    });


    const rangeInput = document.getElementById('rangeInput');
    rangeInput.addEventListener('input', function () {
        nilaiSlider = rangeInput.value;
        document.getElementById('selectedNumber').textContent = nilaiSlider
    });
    
    editModal.addEventListener('show.bs.modal', function (event) {
        const slider = document.getElementById('edit2');
        document.getElementById('weight').textContent = slider.value;
        slider.addEventListener('input', function () {
            nilaiSlider = slider.value;
            document.getElementById('weight').textContent = nilaiSlider;
        });
    });

    $('#history_tbl').DataTable( {
        dom: 'frtip',
        columnDefs: [ {
            targets: [0,3],
            width: "15%"
        } ],
    } );


    
    
    // $('#normalized_tbl').on('keydown', 'td[contenteditable="true"]', function(event) {
    //     if (event.key === 'Enter') {
    //         event.preventDefault();
    //         var cell = table.cell(this);
    //         var data = cell.data();
    //         var rowIndex = cell.index().row;

    //         var editedData = {
    //             id: table.row(rowIndex).data()[0],
    //             value: data,
    //             csrf_token: table.row(rowIndex).data()[4]
    //         };

    //         $.ajax({
    //             type: 'POST',
    //             url: '/edit_normalisasi/',
    //             contentType: 'application/json;charset=UTF-8',
    //             headers: {
    //                 'X-CSRFToken': editedData.csrf_token
    //             },
    //             data: JSON.stringify(editedData),
    //             success: function(response) {
    //                 // console.log(response);
    //                 // window.location.href = '/tes';
    //                 console.log(response);
    //             },
    //             error: function(error) {
    //                 console.error('Error updating data:', error);
    //             }
    //         });
    //     }
    // });

} );

function showPasswd(id) {
    const x = document.getElementById(id);
    const y = document.getElementById("showpass");
    if (x.type === "password") {
        x.type = "text";
        y.className = "bi bi-eye-fill";
    } else {
        x.type = "password";
        y.className = "bi bi-eye-slash-fill";
    }
}

function verifyPassword(id)
{ 
    const passwd = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{6,20}$/;
    const x = document.getElementById(id);
    const y = document.getElementById("description");
    const z = document.getElementById("submit");

    if (!x.value.match(passwd))
    { 
        y.innerHTML = "<small>*Masukkan 6-20 karakter password, berisi kombinasi angka, huruf kecil dan huruf kapital.</small>"
    } else {
        y.innerHTML = ""
        z.disabled = false;
    }
}

function CheckPassword(id) 
{ 
    const passwd = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{6,20}$/;
    const showpass = document.getElementById("showpass")
    const x = document.getElementById(id);

    if(x.value.length >= 6 && x.value.length <= 20) 
    { 
        document.getElementById("rule1").innerHTML = "<i class='bi bi-check' style='color:green;'></i>"
    } else {
        document.getElementById("rule1").innerHTML = "<i class='bi bi-x' style='color:red;'></i>"
    }

    if(x.value.match(passwd))
    { 
        document.getElementById("rule2").innerHTML = "<i class='bi bi-check' style='color:green;'></i>"
    }
}

function comparePassword()
{ 
    const x = document.getElementById("new_password");
    const y = document.getElementById("password_confirmation");
    const z = document.getElementById("change_password");

    if(x.value == y.value)
    {
        document.getElementById("rule3").innerHTML = "<i class='bi bi-check' style='color:green;'></i>"
        z.disabled = false;
    }
}

function generatePswd(id) {
    var chars = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    var passwordLength = 12;
    var password = "";

    for (var i = 0; i <= passwordLength; i++) {
        var randomNumber = Math.floor(Math.random() * chars.length);
        password += chars.substring(randomNumber, randomNumber +1);
    }
    document.getElementById(id).value = password;
}

function confirmDelete(url)
{
    Swal.fire({
        title: 'Anda Yakin?',
        text: "Anda tidak bisa mengembalikan data setelah dihapus!",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#d33',
        cancelButtonColor: '#000',
        confirmButtonText: 'Yakin dong...',
        cancelButtonText: 'Gak jadi deh!'
      }).then((result) => {
        if (result.isConfirmed) {
            location = url;
        }
      })
}

// Populate data to edit Modal

function showElement(id) {
    var el = document.getElementById(id);
    el.style.display = "block";
}

const editModal = document.getElementById('editModal');
const editForm = document.getElementById('editForm');
const editId = document.getElementById('editId');
const edit1 = document.getElementById('edit1');
const edit2 = document.getElementById('edit2');
const edit3 = document.getElementById('edit3');
const edit4 = document.getElementById('edit4');

editModal.addEventListener('show.bs.modal', function (event) {
    const button = event.relatedTarget;
    const dataUrl = button.getAttribute('data-bs-url');
    const dataId = button.getAttribute('data-bs-id');
    const data1 = button.getAttribute('data-bs-1');
    const data2 = button.getAttribute('data-bs-2');
    const data3 = button.getAttribute('data-bs-3');
    const data4 = button.getAttribute('data-bs-4');


    editForm.action = dataUrl;
    editId.value = dataId;
    edit1.value = data1;
    edit2.value = data2;
    edit3.value = data3;
    edit4.value = data4;

    const modalBodyInput = editModal.querySelector('.modal-body form');
    modalBodyInput.appendChild(editId);
});

editModal.addEventListener('hide.bs.modal', function (event) {
    editId.remove();
});

function resetInput() {
    const x = document.getElementById("search-bar");
    const y = document.getElementById("search-button");
    const z = document.getElementById("search-icon");

    if (x.value.length > 0) {
        z.className = "bi bi-x-lg";
    }
}

function readyToSearch() {
    const x = document.getElementById("search-bar");
    const y = document.getElementById("search-button");
    const z = document.getElementById("search-icon");

    if (z.className == "bi bi-x-lg") {
        y.type = "reset";
        z.className = "bi bi-search";
    } else {
        y.type = "submit";
        z.className = "bi bi-x-lg";}
}

function markAsVisited(element) {
    element.style.color = '#FF6347'; // Ganti dengan warna yang diinginkan
    element.removeAttribute('onmouseover'); // Menghapus event onmouseover setelah diklik
    element.removeAttribute('onmouseout');  // Menghapus event onmouseout setelah diklik
}

function autoResize() {
    // Mendapatkan elemen textarea
    var textarea = document.getElementById('text-area');

    // Menghitung jumlah karakter dalam textarea
    var characters = textarea.value.length;

    // Mengatur ukuran font berdasarkan jumlah karakter
    // Anda dapat mengatur formula yang sesuai dengan kebutuhan Anda
    var fontSize = Math.max(12, 30 - characters * 0.1) + 'px';

    // Mengatur ukuran font pada textarea
    textarea.style.fontSize = fontSize;
}

window.onload = autoResize;